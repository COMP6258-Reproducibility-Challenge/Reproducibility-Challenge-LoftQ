
from transformers.hf_argparser import HfArgumentParser

from arguments import BaseArguments, ModelArguments, DataArguments, TrainingArguments

from utils import load_raw_dataset
import model_utils
import classification_utils

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

if __name__ == "__main__":
    parser = HfArgumentParser((BaseArguments, ModelArguments, DataArguments, TrainingArguments))
    base_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    raw_data = load_raw_dataset(data_args.data_name, data_args.task_name)
    num_labels, labels = classification_utils.count_labels(raw_data)
    
    model_name = model_args.model_name_or_path
    
    if not base_args.from_saved:
    
        model, tokenizer, target_modules, excluded_modules = model_utils.load_base_model(model_name, model_args.token, num_labels)   
        base_total = model_utils.count_total_parameters(model)
        base_trainable = model_utils.count_trainable_parameters(model)
        print(f"Base - Total parameters: {base_total:,}")
        print(f"Base - Trainable parameters: {base_trainable:,} ({100*base_trainable/base_total:.2f}%)") 
        model = model_utils.quantize_model(model, model_args, target_modules, excluded_modules)
        model_utils.save_quantized_model(model, tokenizer, base_args.save_dir, model_args)
    else:
        model_class = model_utils.get_base_class(model_name)
        model, tokenizer, target_modules, excluded_modules = model_utils.load_loftq_model(model_class, model_args, base_args.save_dir)
        # test_layer_quantization(model)
    
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            # Keep all LoRA adapters trainable
            param.requires_grad = True
        elif any(excluded in name.lower() for excluded in excluded_modules):
            # Keep excluded layers trainable (full precision)
            param.requires_grad = True
        else:
            # Freeze everything else (quantized backbone)
            param.requires_grad = False

    quant_total = model_utils.count_total_parameters(model)
    quant_trainable = model_utils.count_trainable_parameters(model)
    print(f"LoftQ - Total parameters: {quant_total:,}")
    print(f"LoftQ - Trainable parameters: {quant_trainable:,} ({100*quant_trainable/quant_total:.2f}%)")
    if not training_args.no_train:
        if not training_args.train_small is None and training_args.train_small == True:
            train_small = classification_utils.create_reduced_dataset(raw_data["train"], labels, num_examples=100)
            val_small = classification_utils.create_reduced_dataset(raw_data["validation_matched"], labels, num_examples=3)
            val_mm_small = classification_utils.create_reduced_dataset(raw_data["validation_mismatched"], labels, num_examples=3)
            raw_data['train'] = train_small
            raw_data['validation_matched'] = val_small
            raw_data['validation_mismatched'] = val_mm_small
            classification_utils.train(model, tokenizer, model_args, data_args, training_args, raw_data)
        else:
            classification_utils.train(model, tokenizer, model_args, data_args, training_args, raw_data)