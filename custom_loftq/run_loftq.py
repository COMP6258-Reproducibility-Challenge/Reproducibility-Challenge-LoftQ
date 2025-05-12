
from transformers.hf_argparser import HfArgumentParser

from arguments import BaseArguments, ModelArguments, DataArguments, TrainingArguments

from utils import load_raw_dataset
import model_utils
import classification_utils
import qa_utils
import summarisation_utils
import clm_utils
from peft import TaskType

from transformers import PreTrainedModel

import torch

if __name__ == "__main__":
    parser = HfArgumentParser((BaseArguments, ModelArguments, DataArguments, TrainingArguments))
    base_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    raw_data = load_raw_dataset(data_args.data_name, data_args.task_name)
    num_labels, labels = classification_utils.count_labels(raw_data)
    
    model_name = model_args.model_name_or_path
    
    if not base_args.from_saved:
        model, task_type, tokenizer, target_modules, excluded_modules = model_utils.load_base_model(model_name, model_args.token, data_args.data_name, num_labels)
        base_total = model_utils.count_total_parameters(model)
        base_trainable = model_utils.count_trainable_parameters(model)
        print(f"Base - Total parameters: {base_total:,}")
        print(f"Base - Trainable parameters: {base_trainable:,} ({100*base_trainable/base_total:.2f}%)") 
        model = model_utils.quantize_model(model, model_args, target_modules, excluded_modules)
        model_utils.save_quantized_model(model, tokenizer, base_args.save_dir, model_args)
    else:
        model_class, task_type = model_utils.get_base_class(model_name,  data_args.data_name)
        model, tokenizer, target_modules, excluded_modules = model_utils.load_loftq_model(model_class, model_args, base_args.save_dir, num_labels=num_labels)

    model_utils.check_model_fits_task(task_type, data_args.data_name.split("/")[-1])
    
    quant_total = model_utils.count_total_parameters(model)
    quant_trainable = model_utils.count_trainable_parameters(model)
    print(f"LoftQ - Total parameters: {quant_total:,}")
    print(f"LoftQ - Trainable parameters: {quant_trainable:,} ({100*quant_trainable/quant_total:.2f}%)")
    
    print(torch.cuda.memory_reserved())
    torch.cuda.empty_cache()
    print(torch.cuda.memory_reserved())
    if not training_args.no_train:
        if task_type == TaskType.SEQ_CLS:
            if not training_args.train_small is None and training_args.train_small == True:
                raw_data['train'] = classification_utils.create_reduced_dataset(raw_data["train"], labels, num_examples=100)
                if data_args.task_name == "mnli":
                    raw_data['validation_matched'] = classification_utils.create_reduced_dataset(raw_data["validation_matched"], labels, num_examples=3)
                    raw_data['validation_mismatched'] = classification_utils.create_reduced_dataset(raw_data["validation_mismatched"], labels, num_examples=3)
                else:
                    raw_data['validation'] = classification_utils.create_reduced_dataset(raw_data["validation"], labels, num_examples=3)
                    
                classification_utils.train(model, tokenizer, model_args, data_args, training_args, raw_data)
            else:
                torch.cuda.reset_peak_memory_stats()
                # do one forward+backward
                classification_utils.train(model, tokenizer, model_args, data_args, training_args, raw_data)
                print(torch.cuda.max_memory_reserved())
        elif task_type == TaskType.QUESTION_ANS:
            if not training_args.train_small is None and training_args.train_small == True:
                raw_data['train'] = qa_utils.create_reduced_dataset(raw_data["train"], num_examples=100)
                raw_data['validation'] = qa_utils.create_reduced_dataset(raw_data["validation"], num_examples=3)
            qa_utils.train(model, tokenizer, model_args, data_args, training_args, raw_data)
        elif task_type == TaskType.SEQ_2_SEQ_LM:
            summarisation_utils.train(model, tokenizer, model_args, data_args, training_args, raw_data)
        elif task_type == TaskType.CAUSAL_LM:
            clm_utils.train(model, tokenizer, model_args, training_args, raw_data)