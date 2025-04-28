from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import os
import json
import copy
import logging
import sys


import evaluate
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset

import transformers
from peft import PeftModel, LoftQConfig, LoraConfig, TaskType, get_peft_model
from dataclasses import dataclass, field
from datasets import load_dataset

import utils
from loftq_custom import LoraLinearLayer
from safetensors.torch import load_file

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

task_to_metrics = {
    "cola": "matthews_correlation",
    "mnli": "accuracy",
    "mrpc": "f1",
    "qnli": "accuracy",
    "qqp": "f1",
    "rte": "accuracy",
    "sst2": "accuracy",
    "stsb": "pearson",
}

@dataclass
class BaseArguments:
    load_dir: Optional[str] = field(
        default="trained_models",
        metadata={"help": "Path to load the trained model."},
    )
    from_saved: Optional[bool] = field(
        default=False,
        metadata={"help": "Load already quantized model"}
    )

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        metadata={"help": "Path to the model."},
    )
    lora_init: bool = field(
        default=False,
        metadata={"help": "True: Use zero and gaussian initialization; False: Load adapters from LoftQ in HF hub."},
    )
    full_precision:  bool = field(
        default=False,
        metadata={"help": "False: Use bitsandbytes Linear4bit, real quantization"
                          "True: Use quantization equivalent fp16/fp32 weights."
                  },
    )
    reduced_rank: int = field(
        default=32,
        metadata={"help": "Quantized Rank"},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoftQ does not require this config. Used for QLoRA."},
    )
    token: Optional[str] = field(
        default=None,
        metadata={"help": "HF token to access to private models, e.g., meta-llama"},
    )
    num_iter: Optional[int] = field(
        default=5,
        metadata={"help": "The number of iterations"},
    )
    int_bit: Optional[int] = field(
        default=4,
        metadata={"help": "Integer bit"},
    )
    decompose: Optional[bool] = field(
        default=False,
        metadata={"help": "whether decompose"},
    )
    quant_embedding: Optional[bool] = field(
        default=False,
        metadata={"help": "Quantize embeddings"},
    )
    quant_method: Optional[str] = field(
        default="uniform",
        metadata={"help": "Quantize method: uniform or nf"},
    )
    loftq: Optional[bool] = field(
        default=False,
        metadata={"help": "Quantize method: uniform or nf"},
    )
    qlora: Optional[bool] = field(
        default=False,
        metadata={"help": "Quantize method: uniform or nf"},
    )

@dataclass
class DataArguments:
    data_name: str = field(
        default="gsm8k",
        metadata={"help": "Dataset name."}
    )
    task_name: str = field(
        default="main",
        metadata={"help": "Dataset name."}
    )
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    expt_name: str = field(
        default="default",
        metadata={"help": "Experiment name"},
    )
    
def load_raw_dataset(dataset_name, task_name):
    logger.warning("Loading raw dataset")
    raw_data = load_dataset(dataset_name, task_name)
    train_set = raw_data["train"]
    labels = set([entry['label'] for entry in train_set])
    num_labels = len(labels)
    return raw_data, labels, num_labels

def get_target_blocked_modules(model_args):
    if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]

    elif any(name in model_args.model_name_or_path.lower() for name in ["bart", "t5"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "fc1", "fc2", "out_proj"]

    elif any(name in model_args.model_name_or_path.lower() for name in ["deberta", "roberta", "bert"]):
        target_modules = ['query', 'key', 'value',
                  'q_proj', 'k_proj', 'v_proj',
                  'query_proj', 'key_proj', 'value_proj',
                  'out_proj', 'dense', 'attention', 'fc1', 'fc2']
        # target_modules = [
        #     "query", "key", "value", "q_proj",
        #     "k_proj", "v_proj", "query_proj",
        #     "key_proj", "value_proj", "out_proj",
        #     "dense", "output.dense", "self.query_proj",
        #     "self.key_proj", "self.value_proj"
        # ]
    else:
        raise NotImplementedError("Other models not supported yet.")
    
    blocked_modules = ['pooler', 'classifier', 'LayerNorm']
    
    return target_modules, blocked_modules

def load_loftq_model(model_class, model_args, save_dir):
    """
    Load a model with LoraLinearLayer modules.
    """
    model_dir = get_model_dir(save_dir, model_args)
    # Load the model architecture and basic parameters
    if issubclass(model_class, transformers.PreTrainedModel):
        # For HuggingFace models
        model = model_class.from_pretrained(model_dir)
    else:
        # For regular PyTorch models
        model = model_class(**(model_args if model_args else {}))
        model.load_state_dict(torch.load(os.path.join(save_dir, f"loftq_model.pt")), strict=False)
        
    module_mapping = torch.load(os.path.join(model_dir, f"loftq_mapping.pt"))
    
    # First replace all mapped linear layers with LoFTQLayer
    for name, module_info in module_mapping.items():
        if module_info["type"] == "LoraLinearLayer":
            parent_name = ".".join(name.split(".")[:-1])
            module_name = name.split(".")[-1]
            
            if not parent_name:
                parent = model
            else:
                parent = model.get_submodule(parent_name)
            
            # Get the original linear layer
            try:
                original_module = parent.get_submodule(module_name)
                
                # Create a new LoFTQLayer
                loftq_layer = LoraLinearLayer(
                    base_layer=original_module if isinstance(original_module, torch.nn.Linear) else torch.nn.Linear(
                        module_info["in_features"],
                        module_info["out_features"],
                        bias=hasattr(original_module, 'bias') and original_module.bias is not None
                    ),
                    quantization_bits=module_info["quantization_bits"],
                    reduced_rank=module_info["reduced_rank"],
                    num_iters=module_info["num_iters"],
                    quantization_method=module_info["quantization_method"]
                )
                
                # Replace the module
                setattr(parent, module_name, loftq_layer)
            except AttributeError:
                print(f"Warning: Module {module_name} not found in {parent_name}")
                
    loftq_state_dict = torch.load(os.path.join(model_dir, f"loftq_weights.pt"))
        
    with torch.no_grad():
        for name, param in loftq_state_dict.items():
            if isinstance(param, torch.Tensor):
                # Get the parameter from the model and assign the loaded value
                module_path = name.rsplit(".", 1)[0]  # Get module path without parameter name
                param_name = name.split(".")[-1]  # Get parameter name
                
                try:
                    module = model.get_submodule(module_path)
                    if "base_layer" in param_name:
                        getattr(module, 'base_layer').weight.copy_(param)
                    elif hasattr(module, param_name):
                        getattr(module, param_name).weight.copy_(param)
                except AttributeError as e:
                    print(f"Warning: Could not set parameter {name}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    
    return model, tokenizer

# def load_model(model_args, load_dir):
#     logger.warning("Loading quantized model")
#     model_dir = get_model_dir(load_dir, model_args)

#     model = transformers.AutoModelForSequenceClassification.from_pretrained(model_dir)
#     tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    
#     loftq_config_path = os.path.join(model_dir, "loftq_config.json")
#     if os.path.exists(loftq_config_path):
#         with open(loftq_config_path, "r") as f:
#             loftq_config = json.load(f)
        
#         # Reapply LoftQ structure
#         model_args.loftq = loftq_config["loftq"]
#         model_args.qlora = loftq_config["qlora"]
#         utils.replace_module(
#             model, 
#             allow_name=loftq_config["allow_name"],
#             reduced_rank=loftq_config["reduced_rank"],
#             decomposition=False,  # Don't reapply quantization
#             int_bit=loftq_config["int_bit"],
#             args=model_args
#         )
    
#     target_modules, blocked_modules = get_target_blocked_modules(model_args)
    
#     return model, tokenizer, target_modules, blocked_modules

def compute_metrics(eval_pred):
    metric = evaluate.load(data_args.data_name, data_args.task_name)
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

def test_loftq_on_mnli(
    model,
    tokenizer,
    model_args,
    data_args,
    training_args,
    raw_dataset,
    output_dir: str = "./results",
    batch_size: int = 32,
    max_length: int = 256,
):
    """
    Test a LoFTQ model on MNLI using the Transformers Trainer.
    
    Args:
        model_path: Path to the saved LoFTQ model
        output_dir: Directory to save test results
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length
        tokenizer_name: Name of the tokenizer to use
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model.to(device)
    
    # Load MNLI dataset - both matched and mismatched validation sets
    print("Loading MNLI dataset")
    mnli_matched = load_dataset("glue", "mnli", split="validation_matched")
    mnli_mismatched = load_dataset("glue", "mnli", split="validation_mismatched")
    
    # Preprocessing function
    def preprocess_function(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
    
    # Preprocess datasets
    print("Preprocessing datasets")
    mnli_matched = mnli_matched.map(preprocess_function, batched=True)
    mnli_mismatched = mnli_mismatched.map(preprocess_function, batched=True)
    
    # Load accuracy metric
    accuracy_metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return accuracy_metric.compute(predictions=predictions, references=labels)
    
    # Set up training arguments for evaluation
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=batch_size,
        remove_unused_columns=True,
    )
    
    # Create trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
    )
    
    # Run evaluation on matched validation set
    print("Evaluating on MNLI matched validation set")
    matched_results = trainer.evaluate(eval_dataset=mnli_matched)
    print(f"MNLI matched results: {matched_results}")
    
    # Run evaluation on mismatched validation set
    print("Evaluating on MNLI mismatched validation set")
    mismatched_results = trainer.evaluate(eval_dataset=mnli_mismatched)
    print(f"MNLI mismatched results: {mismatched_results}")
    
    # Combine results
    all_results = {
        "mnli_matched": matched_results,
        "mnli_mismatched": mismatched_results
    }
    
    # Print summary
    print("\nSummary:")
    print(f"MNLI matched accuracy: {matched_results['eval_accuracy']:.4f}")
    print(f"MNLI mismatched accuracy: {mismatched_results['eval_accuracy']:.4f}")
    
    return all_results

def test(model, tokenizer, model_args, data_args, training_args, raw_dataset):
    logger.warning("Preparing to train model")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")
    
    if (data_args.task_name == "mnli"): 
        test_matched = raw_dataset['test_matched']
        test_mismatched = raw_dataset['test_matched']
        
        test_matched_processed = test_matched.map(
            preprocess_function,
            batched=True,
            remove_columns=["premise", "hypothesis", "idx"]
        )
        test_mismatched_processed = test_mismatched.map(
            preprocess_function,
            batched=True,
            remove_columns=["premise", "hypothesis", "idx"]
        )
    
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
    )
    
    # Get predictions for matched test set
    logger.info("Generating predictions for matched test set")
    matched_preds = trainer.predict(test_matched_processed)
    matched_pred_ids = np.argmax(matched_preds.predictions, axis=1)
    
    # Get predictions for mismatched test set
    logger.info("Generating predictions for mismatched test set")
    mismatched_preds = trainer.predict(test_mismatched_processed)
    mismatched_pred_ids = np.argmax(mismatched_preds.predictions, axis=1)
    
    # Map predictions to labels
    id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    matched_pred_labels = [id2label[id] for id in matched_pred_ids]
    mismatched_pred_labels = [id2label[id] for id in mismatched_pred_ids]
    
    # Create DataFrames for results
    matched_results = pd.DataFrame({
        "idx": test_matched["idx"],
        "prediction": matched_pred_ids,
        "label": matched_pred_labels
    })
    
    mismatched_results = pd.DataFrame({
        "idx": test_mismatched["idx"],
        "prediction": mismatched_pred_ids,
        "label": mismatched_pred_labels
    })
    
    # Save results
    output_dir = "./test_results"
    matched_results.to_csv(os.path.join(output_dir, "mnli_matched_results.csv"), index=False)
    mismatched_results.to_csv(os.path.join(output_dir, "mnli_mismatched_results.csv"), index=False)
    
    logger.info(f"Test predictions saved to {output_dir}")
    
    # Save in GLUE submission format
    matched_submission = pd.DataFrame({
        "index": test_matched["idx"],
        "prediction": matched_pred_labels
    })
    
    mismatched_submission = pd.DataFrame({
        "index": test_mismatched["idx"],
        "prediction": mismatched_pred_labels
    })
    
    matched_submission.to_csv(os.path.join(output_dir, "mnli-m.tsv"), sep="\t", index=False)
    mismatched_submission.to_csv(os.path.join(output_dir, "mnli-mm.tsv"), sep="\t", index=False)
    
    logger.info(f"GLUE submission files saved to {output_dir}")
    
    return {
        "matched_results": matched_results,
        "mismatched_results": mismatched_results
    }

def get_model_dir(load_dir, model_args):
    model_name = model_args.model_name_or_path.split("/")[-1] + f"-{model_args.int_bit}bit" + f"-{model_args.reduced_rank}rank"
    return os.path.join(load_dir, model_name)

def preprocess_function(examples):
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        padding="max_length",
        truncation=True,
        max_length=256
    )
    
def create_reduced_dataset(dataset, num_examples=1000, seed=42):
    """Create a smaller subset of the dataset for quick testing."""
    # Set seed for reproducibility
    import numpy as np
    np.random.seed(seed)
    
    indices = [i for i in range(len(dataset))]
    
    # Shuffle the indices
    np.random.shuffle(indices)
    
    reduced_indices = indices[:num_examples]
    
    # Create the reduced dataset
    reduced_dataset = dataset.select(reduced_indices)
    print(f"Created reduced dataset with {len(reduced_dataset)} examples")
    
    return reduced_dataset

def get_base_class(model_args):
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    model_type = config.model_type
    
    if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon"]):
        return transformers.AutoModelForCausalLM

    elif any(name in model_args.model_name_or_path.lower() for name in ["bart", "t5"]):
        return transformers.AutoModelForSeq2SeqLM

    elif any(name in model_args.model_name_or_path.lower() for name in ["deberta", "roberta", "bert"]):
        print(model_type)
        if model_type == "bert":
            from transformers import BertForSequenceClassification as ModelClass
        elif "deberta" in model_type:
            from transformers import DebertaV2ForSequenceClassification as ModelClass
        elif model_type == "roberta":
            from transformers import RobertaForSequenceClassification as ModelClass
        return ModelClass
    else:
        raise NotImplementedError("Other models not supported yet.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = transformers.HfArgumentParser((BaseArguments, ModelArguments, DataArguments, TrainingArguments))
    base_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    raw_data, labels, num_labels = load_raw_dataset(data_args.data_name, data_args.task_name)
    # test_small = create_reduced_dataset(raw_data['test_matched'], 10)
    # test_mm_small = create_reduced_dataset(raw_data['test_mismatched'], 10)
    # raw_data['test_matched'] = test_small
    # raw_data['test_mismatched'] = test_mm_small
    
    model_class = get_base_class(model_args)
    model, tokenizer = load_loftq_model(model_class, model_args, base_args.load_dir)

    test_loftq_on_mnli(model, tokenizer, model_args, data_args, training_args, raw_data)
    # test(model, tokenizer, model_args, data_args, training_args, raw_data)
    # # print("\nApplying PEFT LoRA implementation...")
    # peft_model = transformers.AutoModelForSequenceClassification.from_pretrained(
    #     model_args.model_name_or_path,
    #     num_labels=3,
    # )
    
    # peft_config = LoraConfig(
    #     task_type=TaskType.SEQ_CLS,
    #     inference_mode=False,
    #     r=model_args.reduced_rank,  # Match rank with LoftQ
    #     lora_alpha=model_args.reduced_rank,  # Common setting is alpha=r
    #     lora_dropout=0.1,
    #     # Target modules similar to what replace_module targets
    #     target_modules=[
    #         "query_proj", 
    #         "key_proj", 
    #         "value_proj", 
    #         "dense", 
    #         "output.dense",
    #         "attention.self.query",
    #         "attention.self.key",
    #         "attention.self.value",
    #         "intermediate.dense"
    #     ]
    # )

    # # Apply PEFT configuration
    # peft_model = get_peft_model(peft_model, peft_config)
    
    # peft_total = count_total_parameters(peft_model)
    # peft_trainable = count_trainable_parameters(peft_model)
    # print(f"PEFT LoRA - Total parameters: {peft_total:,}")
    # print(f"PEFT LoRA - Trainable parameters: {peft_trainable:,} ({100*peft_trainable/peft_total:.2f}%)")
    
    # train(peft_model, tokenizer, model_args, data_args, training_args)
    