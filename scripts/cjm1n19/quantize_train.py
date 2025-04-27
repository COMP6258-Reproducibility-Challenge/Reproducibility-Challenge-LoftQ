from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import os
import json
import copy
import logging
import sys


import evaluate
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset

import transformers
from peft import PeftModel, LoftQConfig, LoraConfig, TaskType, get_peft_model
from dataclasses import dataclass, field
from datasets import load_dataset
import utils
import loftq_custom

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
    save_dir: Optional[str] = field(
        default="quantized_model",
        metadata={"help": "Path to save the quantized model."},
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
    train_small: Optional[bool] = field(
        default=False,
        metadata={"help": "Experiment name"},
    )
    
def load_raw_dataset(dataset_name, task_name):
    logging.warning("Loading raw dataset")
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

def load_base_model(model_args, num_labels):
    logging.warning("Loading base model")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon"]):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            token=model_args.token,
            trust_remote_code=True,
            device_map="auto",
        )
        task_type = TaskType.CAUSAL_LM

    elif any(name in model_args.model_name_or_path.lower() for name in ["bart", "t5"]):
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, token=model_args.token)
        task_type = TaskType.SEQ_2_SEQ_LM

    elif any(name in model_args.model_name_or_path.lower() for name in ["deberta", "roberta", "bert"]):
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            token=model_args.token,
            num_labels=num_labels
        )
        task_type = TaskType.SEQ_CLS
    else:
        raise NotImplementedError("Other models not supported yet.")
    
    target_modules, blocked_modules = get_target_blocked_modules(model_args)
    
    return model, tokenizer, target_modules, blocked_modules

def quantize_model(model, model_args, allow_name, block_name):
    logging.warning("Quantizing model, this may take a while")
    # utils.replace_module(model,
    #         allow_name=allow_name,
    #         block_name=block_name,
    #         reduced_rank=model_args.reduced_rank,
    #         decomposition=model_args.decompose,
    #         quant_method=model_args.quant_method,
    #         int_bit=model_args.int_bit,
    #         args=model_args,
    #     )
    
    model = loftq_custom.convert_linear_layer(
        model,
        quantization_bits=model_args.int_bit,
        # target_modules=allow_name,
        rank=model_args.reduced_rank,
        quantization_method=model_args.quant_method,
        target_modules=["query_proj", "key_proj", "value_proj", "dense"]
    )
    return model

def save_quantized_model(model, tokenizer, save_path, model_args):
    logging.warning("Saving quantized model")
    os.makedirs(save_path, exist_ok=True)
    model_dir = get_model_dir(save_path, model_args)
    
    # if hasattr(model, 'config'):
    #     model.config.save_pretrained(model_dir)
        
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

def save_quant_config(save_dir, model_args, target_modules):
    with open(os.path.join(get_model_dir(save_dir, model_args), "loftq_config.json"), "w") as f:
        json.dump({
            "loftq": model_args.loftq,
            "qlora": model_args.qlora,
            "int_bit": model_args.int_bit,
            "reduced_rank": model_args.reduced_rank,
            "allow_name": target_modules
        }, f)

def load_quantized_model(model_args, save_dir):
    logging.warning("Loading quantized model")
    model_dir = get_model_dir(save_dir, model_args)

    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    
    loftq_config_path = os.path.join(model_dir, "loftq_config.json")
    if os.path.exists(loftq_config_path):
        with open(loftq_config_path, "r") as f:
            loftq_config = json.load(f)
        
        # Reapply LoftQ structure
        model_args.loftq = loftq_config["loftq"]
        model_args.qlora = loftq_config["qlora"]
        utils.replace_module(
            model, 
            allow_name=loftq_config["allow_name"],
            reduced_rank=loftq_config["reduced_rank"],
            decomposition=False,  # Don't reapply quantization
            int_bit=loftq_config["int_bit"],
            args=model_args
        )
    
    target_modules, blocked_modules = get_target_blocked_modules(model_args)
    
    return model, tokenizer, target_modules, blocked_modules

def compute_metrics(eval_pred):
    metric = evaluate.load(data_args.data_name, data_args.task_name)
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

def create_reduced_dataset(dataset, num_examples=1000, seed=42):
    """Create a smaller subset of the dataset for quick testing."""
    # Set seed for reproducibility
    import numpy as np
    np.random.seed(seed)
    
    # Get a balanced subset
    indices = []
    for label in [0, 1, 2]:  # MNLI has 3 classes
        label_indices = [i for i, l in enumerate(dataset["label"]) if l == label]
        selected = np.random.choice(label_indices, min(num_examples // 3, len(label_indices)), replace=False)
        indices.extend(selected)
    
    # Shuffle the indices
    np.random.shuffle(indices)
    
    # Create the reduced dataset
    reduced_dataset = dataset.select(indices)
    print(f"Created reduced dataset with {len(reduced_dataset)} examples")
    
    return reduced_dataset

def train(model, tokenizer, model_args, data_args, training_args, raw_dataset):
    logging.warning("Preparing to train model")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")
    
    tokenized_datasets = raw_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["premise", "hypothesis", "idx"]
    )
    
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation_matched"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Training model...")
    if (not training_args.resume_from_checkpoint is None):
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()

    # Step 10: Evaluate on validation sets
    print("Evaluating model...")
    results_matched = trainer.evaluate(tokenized_datasets["validation_matched"])
    results_mismatched = trainer.evaluate(tokenized_datasets["validation_mismatched"])

    print(f"Validation Matched Results: {results_matched}")
    print(f"Validation Mismatched Results: {results_mismatched}")

    # Step 11: Save the fine-tuned model
    print("Saving final fine-tuned model...")
    model_dir = get_model_dir("trained_models", model_args)
    trainer.save_model(model_dir)

    print("Process completed!")

def count_trainable_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_parameters(model):
    """Count total parameters in a model"""
    return sum(p.numel() for p in model.parameters())

def get_model_dir(save_dir, model_args):
    model_name = model_args.model_name_or_path.split("/")[-1] + f"-{model_args.int_bit}bit" + f"-{model_args.reduced_rank}rank"
    return os.path.join(save_dir, model_name)

def preprocess_function(examples):
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        padding="max_length",
        truncation=True,
        max_length=256
    )
    
def test_layer_quantization(model):
    """Test the quantization of each layer independently."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(attr_str in name for attr_str in ['query', 'key', 'value']):
            print(f"Testing quantization for {name}")
            
            # Save original weights
            original_weight = module.weight.clone()
            
            # Apply quantization to this layer only
            reduced_rank = 8  # Use a small rank for testing
            L, R = utils.low_rank_decomposition(original_weight, reduced_rank)
            quant_w = utils.weight_quant_fn(original_weight - L @ R, num_bits=4, quant_method='uniform')
            
            # Check error
            error = torch.norm(original_weight - quant_w - L @ R).item()
            print(f"  Quantization error: {error:.4f}")
            
            # Check weight statistics
            print(f"  Original weight range: [{original_weight.min().item():.4f}, {original_weight.max().item():.4f}]")
            print(f"  Quantized weight range: [{quant_w.min().item():.4f}, {quant_w.max().item():.4f}]")
            print(f"  Low-rank component range: [{(L @ R).min().item():.4f}, {(L @ R).max().item():.4f}]")
            
if __name__ == "__main__":
    parser = transformers.HfArgumentParser((BaseArguments, ModelArguments, DataArguments, TrainingArguments))
    base_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    raw_data, labels, num_labels = load_raw_dataset(data_args.data_name, data_args.task_name)
    
    if not base_args.from_saved:
    
        model, tokenizer, target_modules, blocked_modules = load_base_model(model_args, num_labels)    
        model = quantize_model(model, model_args, target_modules, blocked_modules)
        save_quantized_model(model, tokenizer, base_args.save_dir, model_args)
        save_quant_config(base_args.save_dir, model_args, target_modules)
    else:
        model, tokenizer, target_modules, blocked_modules = load_quantized_model(model_args, base_args.save_dir)
        # test_layer_quantization(model)
    
    for name, param in model.named_parameters():
    # Keep trainable: 1) all LoRA adapters and 2) non-LoRA classifier parameters
        if "lora" not in name.lower() and "classifier" not in name.lower() and "pooler" not in name.lower():
            param.requires_grad = False
        else:
            param.requires_grad = True

    quant_total = count_total_parameters(model)
    quant_trainable = count_trainable_parameters(model)
    print(f"LoftQ - Total parameters: {quant_total:,}")
    print(f"LoftQ - Trainable parameters: {quant_trainable:,} ({100*quant_trainable/quant_total:.2f}%)")

    # if not training_args.train_small is None and training_args.train_small == True:
    #     dataset = load_dataset("glue", "mnli")
    #     train_small = create_reduced_dataset(dataset["train"], num_examples=100)
    #     val_small = create_reduced_dataset(dataset["validation_matched"], num_examples=3)
    #     val_mm_small = create_reduced_dataset(dataset["validation_mismatched"], num_examples=3)
    #     raw_data['train'] = train_small
    #     raw_data['validation_matched'] = val_small
    #     raw_data['validation_mismatched'] = val_mm_small
    #     train(model, tokenizer, model_args, data_args, training_args, raw_data)
    # else:
    #     train(model, tokenizer, model_args, data_args, training_args, raw_data)
    
    print("\nApplying PEFT LoRA implementation...")
    peft_model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=3,
    )
    loftq_config = LoftQConfig(loftq_bits=model_args.int_bit, loftq_iter=model_args.num_iter)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=model_args.reduced_rank,  # Match rank with LoftQ
        lora_alpha=model_args.reduced_rank,  # Common setting is alpha=r
        init_lora_weights="loftq",
        loftq_config=loftq_config,
        target_modules=["query_proj", "key_proj", "value_proj", "dense"]
    )

    # Apply PEFT configuration
    peft_model = get_peft_model(peft_model, peft_config)
    
    peft_total = count_total_parameters(peft_model)
    peft_trainable = count_trainable_parameters(peft_model)
    print(f"PEFT LoRA - Total parameters: {peft_total:,}")
    print(f"PEFT LoRA - Trainable parameters: {peft_trainable:,} ({100*peft_trainable/peft_total:.2f}%)")
    
    # train(peft_model, tokenizer, model_args, data_args, training_args)
    