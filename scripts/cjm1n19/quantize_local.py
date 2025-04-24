from dataclasses import dataclass, field
from typing import Optional

import transformers
import torch
from peft import PeftModel, LoftQConfig, LoraConfig, TaskType, get_peft_model

import utils
import os
import json

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

def quantize():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    # model = PeftModel.from_pretrained(
    #         model,
    #         model_args.adapter_name_or_path,
    #         is_trainable=True,
    #         token=model_args.token,
    # )

    original_model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        low_cpu_mem_usage=True,
        # torch_dtype=torch.bfloat16
    )

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        low_cpu_mem_usage=True,
        # torch_dtype=torch.bfloat16
    )

    task_type = TaskType.SEQ_CLS
    allow_name = [
        "query", "key", "value", "q_proj",
        "k_proj", "v_proj", "query_proj",
        "key_proj", "value_proj", "out_proj",
        "dense", "output.dense", "self.query_proj",
        "self.key_proj", "self.value_proj"
    ]
    block_name = ['pooler', 'classifier', 'LayerNorm']

    utils.replace_module(model,
                            allow_name=allow_name,
                            block_name=block_name,
                            reduced_rank=model_args.reduced_rank,
                            decomposition=model_args.decompose,
                            quant_method=model_args.quant_method,
                            int_bit=model_args.int_bit,
                            args=model_args,
                        )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        padding_side="right",
        use_fast=False,
    )

    save_directory = "./quantized_model"
    tokenizer.save_pretrained(save_directory)
    save_quantized_model(model, save_directory)
    create_peft_config(model_args)
    verify_quantization(original_model, model, tokenizer)

def save_quantized_model(model, save_path):
    """Save quantized model components separately for reliable loading"""
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save model configuration
    if hasattr(model, 'config'):
        model.config.save_pretrained(save_path)
    
    # Save quantization info if needed
    quantization_config = {
        "quantization_method": "custom",
        "bits": 8,  # Or whatever you used
        "custom_method": "manual"
    }
    with open(os.path.join(save_path, "quantization_config.json"), "w") as f:
        json.dump(quantization_config, f)
    
    # Save state dict
    torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
    
    return save_path

def verify_quantization(original_model, quantized_model, tokenizer, test_input="This is a test input to verify quantization."):
    # Encode input
    inputs = tokenizer(test_input, return_tensors="pt")
    
    # Get outputs
    with torch.no_grad():
        original_output = original_model(**inputs).logits
        quantized_output = quantized_model(**inputs).logits
    
    # Calculate difference
    mae = torch.mean(torch.abs(original_output - quantized_output))
    print(f"Mean Absolute Error: {mae.item()}")
    
    # Simple prediction test
    original_pred = original_output.argmax(-1)
    quantized_pred = quantized_output.argmax(-1)
    print(f"Predictions match: {torch.all(original_pred == quantized_pred).item()}")

def load_quantized():
    model_path = "./quantized_model"
    config = transformers.AutoConfig.from_pretrained(model_path)
    model = transformers.AutoModelForSequenceClassification.from_config(config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def create_peft_config(model_args):
    peft_config = LoraConfig(
        r=model_args.reduced_rank,
        lora_alpha=16,
        target_modules=['query', 'key', 'value',
                  'q_proj', 'k_proj', 'v_proj',
                  'query_proj', 'key_proj', 'value_proj',
                  'out_proj', 'dense', 'attention', 'fc1', 'fc2'],
        bias="none",
        task_type="SEQ_CLS",
        init_lora_weights=False
    )
    
    # Create PEFT directory if needed
    peft_model_path = "./peft_loftq_model"
    os.makedirs(peft_model_path, exist_ok=True)

    # Save the PEFT configuration
    peft_config.save_pretrained(peft_model_path)

if __name__ == "__main__":
    quantize()