import json
import logging
import os
from typing import List, Optional, Tuple, Type, Union, Dict

import warnings
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoConfig,
    PreTrainedModel,
    logging as transformers_logging
)
from peft import TaskType
import torch
import torch.nn as nn

from loftq import LoraLinearLayer, TrueQuantizedLinear, BaseLoftqLinear

from arguments import ModelArguments

def estimate_model_size(original_model, model):
    original_size = sum(p.numel() * p.element_size() for p in original_model.parameters()) / (1024**2)
    print(f"Original Conv2d size: {original_size:.4f} MB")

    quantized_params = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    quantized_buffers = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024**2)
    quantized_size = quantized_params + quantized_buffers
    print(f"Quantized Conv2d size: {quantized_size:.4f} MB")
    print(f"Parameters: {quantized_params:.4f} MB, Buffers: {quantized_buffers:.4f} MB")
    print(f"Compression ratio: {original_size/quantized_size:.2f}x")

def pretty_print_model_args(model_args: ModelArguments):
    return f"Model name: {model_args.model_name_or_path}, Method: {model_args.quant_method}, Rank: {model_args.reduced_rank}, Bits: {model_args.int_bit}, True quantize: {model_args.true_quantization}"

def count_trainable_parameters(model: Union[PreTrainedModel, torch.nn.Module]):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_parameters(model: Union[PreTrainedModel, torch.nn.Module]):
    """Count total parameters in a model"""
    return sum(p.numel() for p in model.parameters())

def get_model_dir(save_dir: str, model_args: ModelArguments) -> str:
    model_name = model_args.model_name_or_path.split("/")[-1] 
    model_details = f"-{model_args.int_bit}bit" + f"-{model_args.reduced_rank}rank" + f"-{model_args.quant_method}_{'true' if model_args.true_quantization else 'sim'}_quant"
    full_name = model_name + model_details
    return os.path.join(save_dir, full_name)

def get_base_class(model_name: str, data_name: str = None) -> Tuple[Type[PreTrainedModel], TaskType]:
    config = AutoConfig.from_pretrained(model_name)
    model_type = config.model_type
    
    if any(name in model_name.lower() for name in ["llama", "mistral", "falcon"]):
        if model_type == "llama":
            from transformers import LlamaForCausalLM as ModelClass
        elif model_type == "mistral":
            from transformers import MistralForCausalLM as ModelClass
        else:
            from transformers import FalconForCausalLM as ModelClass
        return ModelClass, TaskType.CAUSAL_LM

    elif any(name in model_name.lower() for name in ["bart", "t5"]):
        if model_type == "bart":
            from transformers import BartForConditionalGeneration as ModelClass
        else:
            from transformers import T5ForConditionalGeneration as ModelClass
        return ModelClass, TaskType.SEQ_2_SEQ_LM

    elif any(name in model_name.lower() for name in ["deberta", "roberta", "bert"]):
        if data_name == "glue" or data_name == "anli":
            if model_type == "bert":
                from transformers import BertForSequenceClassification as ModelClass
            elif model_type == "roberta":
                from transformers import RobertaForSequenceClassification as ModelClass
            else:
                from transformers import DebertaV2ForSequenceClassification as ModelClass
            return ModelClass, TaskType.SEQ_CLS
        else:
            if model_type == "bert":
                from transformers import BertForQuestionAnswering as ModelClass
            elif model_type == "roberta":
                from transformers import RobertaForQuestionAnswering as ModelClass
            else:
                from transformers import DebertaV2ForQuestionAnswering as ModelClass
            return ModelClass, TaskType.QUESTION_ANS
    else:
        raise NotImplementedError("Other models not supported yet.")

def get_target_excluded_modules(model_name: str) -> Tuple[List[str], List[str]]:
    if any(name in model_name.lower() for name in ["llama", "mistral", "falcon"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]

    elif any(name in model_name.lower() for name in ["bart", "t5"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "fc1", "fc2", "out_proj"]

    elif any(name in model_name.lower() for name in ["deberta", "roberta", "bert"]):
        target_modules = ["query_proj", "key_proj", "value_proj", "dense"]
    else:
        raise NotImplementedError("Other models not supported yet.")
    
    excluded_modules = ['classifier', 'LayerNorm', 'pooler']
    
    return target_modules, excluded_modules

def load_base_model(model_name:str, token: str, data_name: str, num_labels: Optional[int]) -> Tuple[Union[PreTrainedModel, torch.nn.Module], TaskType, AutoTokenizer, List[str], List[str]]:
    logging.warning("Loading base model")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token,
        padding_side="right",
        use_fast=True if data_name == "squad" else False,
        trust_remote_code=True
    )
    if any(name in model_name.lower() for name in ["llama", "mistral", "falcon"]):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            token=token,
            trust_remote_code=True,
            device_map="auto",
        )
        task_type = TaskType.CAUSAL_LM

    elif any(name in model_name.lower() for name in ["bart", "t5"]):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=token)
        task_type = TaskType.SEQ_2_SEQ_LM

    elif any(name in model_name.lower() for name in ["deberta", "roberta", "bert"]):
        if data_name == "glue" or data_name == "anli":
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                token=token,
                num_labels=num_labels
            )
            task_type = TaskType.SEQ_CLS
        else:
            model = AutoModelForQuestionAnswering.from_pretrained(
                model_name,
                token=token
            )
            task_type = TaskType.QUESTION_ANS
    else:
        raise NotImplementedError("Other models not supported yet.")
    
    target_modules, excluded_modules = get_target_excluded_modules(model_name)
    
    return model, task_type, tokenizer, target_modules, excluded_modules
    
def quantize_model(model: Union[PreTrainedModel, torch.nn.Module], model_args: ModelArguments, target_modules: List[str], excluded_modules: List[str]) -> Union[PreTrainedModel, torch.nn.Module]:
    logging.warning("Quantizing model, this may take a while")
    logging.warning(pretty_print_model_args(model_args))
    
    model = convert_linear_layer(
        model,
        quantization_bits=model_args.int_bit,
        rank=model_args.reduced_rank,
        quantization_method=model_args.quant_method,
        target_modules=target_modules,
        excluded_modules=excluded_modules,
        true_quantization=model_args.true_quantization
    )
    
    model = prepare_gradients(model, excluded_modules)
    return model

def save_quantized_model(
    model: Union[PreTrainedModel, torch.nn.Module], 
    tokenizer: AutoTokenizer, 
    save_path: str, 
    model_args: ModelArguments = None
):
    logging.warning("Saving quantized model")
    os.makedirs(save_path, exist_ok=True)
    if model_args is not None:
        model_dir = get_model_dir(save_path, model_args)
    else:
        model_dir = save_path
    
    if isinstance(model, PreTrainedModel):
        # For HuggingFace models
        try:
            model.save_pretrained(model_dir)
        except RuntimeError as e:
            logging.warning(f"Error during save_pretrained: {e}")
            logging.warning("Falling back to custom saving method")
            # Save config separately
            if hasattr(model, 'config'):
                model.config.save_pretrained(model_dir)
    else:
        # For regular PyTorch models
        torch.save(model.state_dict(), os.path.join(model_dir, f"loftq_model.pt"))
        
    loftq_state_dict = {}
    module_mapping = {}
    for name, module in model.named_modules():
        if issubclass(type(module), BaseLoftqLinear):
            module_state = {
                "lora_A": module.lora_A.weight.data,
                "lora_B": module.lora_B.weight.data
            }
            
            if module.has_bias and module.bias is not None:
                module_state["bias"] = module.bias.data
                
            module_mapping[name] = {
                "type": "LoraLinearLayer",
                "in_features": module.in_features,
                "out_features": module.out_features,
                "reduced_rank": module.reduced_rank,
                "quantization_bits": module.quantization_bits,
                'num_iters': module.num_iters,
                'quantization_method': module.quantization_method,
            }
            
            if isinstance(module, LoraLinearLayer):
                module_state["base_layer_weight"] = module.base_layer.weight.data
                module_mapping[name]["type"] = "LoraLinearLayer"
            elif isinstance(module, TrueQuantizedLinear):
                # Save TrueQuantLinearLayer specific parameters
                module_state["qweight"] = module.qweight
                module_state["weight_max"] = module.weight_max
                module_state["weight_shape"] = module.weight_shape
            
                module_mapping[name]["type"] = "TrueQuantLinearLayer"
            
            loftq_state_dict[name] = module_state
        
    torch.save(loftq_state_dict, os.path.join(model_dir, f"loftq_weights.pt"))
    torch.save(module_mapping, os.path.join(model_dir, f"loftq_mapping.pt"))
    
    if model_args is not None:    
        loftq_config = {
            "quant_method": model_args.quant_method,
            "int_bit": model_args.int_bit,
            "reduced_rank": model_args.reduced_rank,
            "true_quantization": model_args.true_quantization
        }
    
        with open(os.path.join(model_dir, f"loftq_config.json"), "w", encoding="utf-8") as writer:
                writer.write(json.dumps(loftq_config, indent=2, sort_keys=True) + "\n")
                writer.close()

    tokenizer.save_pretrained(model_dir)
    
def load_loftq_model(model_class: Type[PreTrainedModel], model_args: ModelArguments, save_dir: str, num_labels: Optional[int]) -> Tuple[Union[PreTrainedModel, torch.nn.Module], AutoTokenizer, List[str], List[str]]:
    """
    Load a model with LoraLinearLayer modules.
    """
    logging.warning("Loading quantized model")
    logging.warning(pretty_print_model_args(model_args))
    
    warnings.filterwarnings("ignore", message=".*Some weights of the model checkpoint.*were not initialized.*")
    transformers_logging.set_verbosity_error()
    
    model_dir = get_model_dir(save_dir, model_args)
    # Load the model architecture and basic parameters
    if issubclass(model_class, PreTrainedModel):
        # For HuggingFace models
        if num_labels is not None:
            model = model_class.from_pretrained(
                model_dir,
                num_labels=num_labels    
            )
        else:
            model = model_class.from_pretrained(
                model_dir
            )
    else:
        # For regular PyTorch models
        # Not actually tested
        model = model_class(**(model_args if model_args else {}))
        model.load_state_dict(torch.load(os.path.join(save_dir, f"loftq_model.pt")), strict=False)
    
    try:
        with open(os.path.join(model_dir, "loftq_config.json"), "r") as fp:
            loftq_config = json.load(fp)
            fp.close()
            for key, value in loftq_config.items():
                if value != getattr(model_args, key):
                    message = f"The loaded quantized model {key} does not match your supplied options:  {value} (saved config) does not match {getattr(model_args, key)}"
                    if model_args.skip_arg_checks:
                        logging.warning(message)
                        logging.warning("Are you sure you wish to continue?")
                    else:
                        raise Exception(message + "- set --skip_arg_checks to ignore")
    except FileNotFoundError:
        logging.warning("No saved loftq config found, using ModelArgs")
    
    module_mapping = torch.load(os.path.join(model_dir, f"loftq_mapping.pt"))
    
    # First replace all mapped linear layers with LoFTQLayer
    for name, module_info in module_mapping.items():
        parent_name = ".".join(name.split(".")[:-1])
        module_name = name.split(".")[-1]
        
        if not parent_name:
            parent = model
        else:
            parent = model.get_submodule(parent_name)
        
        # Get the original linear layer
        try:
            original_module = parent.get_submodule(module_name)
            
            if module_info["type"] == "LoraLinearLayer":
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
            elif module_info["type"] == "TrueQuantLinearLayer":
                loftq_layer = TrueQuantizedLinear(
                    base_layer=original_module if isinstance(original_module, torch.nn.Linear) else torch.nn.Linear(
                        module_info["in_features"],
                        module_info["out_features"],
                        bias=hasattr(original_module, 'bias') and original_module.bias is not None
                    ),
                    quantization_bits=module_info["quantization_bits"],
                    reduced_rank=module_info["reduced_rank"],
                    num_iters=module_info.get("num_iters", 5),
                    quantization_method=module_info["quantization_method"]
                )
            else:
                logging.warning(f"Unknown layer type: {module_info['type']}")
                continue
            
            setattr(parent, module_name, loftq_layer)
        except AttributeError:
            print(f"Warning: Module {module_name} not found in {parent_name}")
                
    loftq_state_dict = torch.load(os.path.join(model_dir, f"loftq_weights.pt"))
    
    with torch.no_grad():
        for module_name, params in loftq_state_dict.items():
            module = model.get_submodule(module_name)
            if isinstance(module, TrueQuantizedLinear) and not "qweight" in params:
                raise Exception("Trying to load true quantized layer without quantized weight")
            if isinstance(module, LoraLinearLayer) and not "base_layer_weight" in params:
                raise Exception("Trying to load simulated quantized layer without dequantized base layer weight")
            
            for param_name, param in params.items():
                if isinstance(param, torch.Tensor):                    
                    try:
                            
                        if param_name == "base_layer_weight":
                            # For LoraLinearLayer
                            module.base_layer.weight.copy_(param)
                        elif param_name in ["qweight", "weight_max", "weight_shape"]:
                            # For TrueQuantLinearLayer buffers
                            module.register_buffer(param_name, param)
                        elif param_name == "bias":
                            # For bias
                            if hasattr(module, 'bias') and module.bias is not None:
                                module.bias.copy_(param)
                        elif param_name == "lora_A":
                            # For lora_A weights
                            module.lora_A.weight.copy_(param)
                        elif param_name == "lora_B":
                            # For lora_B weights
                            module.lora_B.weight.copy_(param)
                        elif hasattr(module, param_name):
                            # For other attributes
                            getattr(module, param_name).copy_(param)
                    except (AttributeError, ValueError) as e:
                        logging.warning(f"Warning: Could not set parameter {name}: {e}")
                    
    target_modules, excluded_modules = get_target_excluded_modules(model_args.model_name_or_path)    

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    model = prepare_gradients(model, excluded_modules)
    
    transformers_logging.set_verbosity_warning() 
    
    return model, tokenizer, target_modules, excluded_modules

def check_model_fits_task(task_type: TaskType, dataset_name: str):
    if task_type == TaskType.SEQ_CLS and dataset_name not in ['glue', 'anli']:
        raise Exception("You are attempting to use a classification model on a non classification task")
    elif task_type == TaskType.QUESTION_ANS and dataset_name not in ["squad"]:
        raise Exception("You are attempting to use a question answering model on a non question answering task")
    elif task_type == TaskType.SEQ_2_SEQ_LM and dataset_name not in ["amazon_reviews_multi", "big_patent", "cnn_dailymail", "orange_sum", "pn_summary", "psc", "samsum", "thaisum", "xglue", "xsum", "wiki_summary", "multi_news"]:
        raise Exception("You are attempting to use a summarisation model on a non summarisation task")
    elif task_type == TaskType.CAUSAL_LM and dataset_name not in ["wikitext-2", "gsm8k"]:
        raise Exception("You are attempting to use a causal model on a non causal task")

def prepare_gradients(model, excluded_modules):
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
    
    return model

def convert_linear_layer(
    model: nn.Module, 
    quantization_bits: int = 4,
    rank: int = 16,
    num_iters: int = 5,
    quantization_method: str = "uniform",
    target_modules: List[str] = ['all-linear'],
    excluded_modules: List[str] = ['classifier'],
    true_quantization: bool = False  # New parameter to enable true quantization
) -> nn.Module:
    """
    Convert a torch.nn.Linear layer in to a custom LoraLinearLayer including quantizing and computing the low-rank decomposition  
    """
    all_linear = len(target_modules) == 1 and target_modules[0] == 'all-linear'
    for name, module in model.named_children():
        
        if any(blocked in name for blocked in excluded_modules):
            continue
        
        if (all_linear and isinstance(module, nn.Linear)) or any(targetted in name for targetted in target_modules):
            print(f"Converting {name} layer with {'true' if true_quantization else 'simulated'} quantization")
            if true_quantization:
                loftq_layer = TrueQuantizedLinear(
                    module,
                    quantization_bits=quantization_bits,
                    reduced_rank=rank,
                    num_iters=num_iters,
                    quantization_method=quantization_method
                )
                
                loftq_layer.quantize(module.weight.data.clone())
            else:
                loftq_layer = LoraLinearLayer(
                    module,
                    quantization_bits=quantization_bits,
                    reduced_rank=rank,
                    num_iters=num_iters,
                    quantization_method=quantization_method
                )
            
                loftq_layer.quantize()
            
            setattr(
                model,
                name,
                loftq_layer    
            )
        else:
            convert_linear_layer(
                module,
                quantization_bits=quantization_bits,
                rank=rank,
                num_iters=num_iters,
                quantization_method=quantization_method,
                target_modules=target_modules,
                excluded_modules=excluded_modules,
                true_quantization=true_quantization
            )
    return model


def requantize_linear_layer(
    model: nn.Module, 
    target_modules: List[str] = ['all-linear'],
    excluded_modules: List[str] = ['classifier'],
    pre_quantized_weights: Dict = None,
    true_quantization: bool = False
) -> nn.Module:
    """
    Convert a torch.nn.Linear layer in to a custom LoraLinearLayer and reapplies precomputed quantized values and low-rank decomposition
    """
    all_linear = len(target_modules) == 1 and target_modules[0] == 'all-linear'
    to_update = []
    for name, module in model.named_modules():
        print(name)
        if any(blocked in name for blocked in excluded_modules):
            continue
        
        if (all_linear and isinstance(module, nn.Linear)) or any(targetted in name for targetted in target_modules):
            to_update.append((name, module))
            print(f"Will convert {name} layer with {'true' if true_quantization else 'simulated'} quantization")
    
    for name, module in to_update:
        if name not in pre_quantized_weights:
            print(f"Warning: No precomputed weights found for {name}, skipping")
            continue
        
        weights_info = pre_quantized_weights[name]
        
        if true_quantization:
            if 'qweight' in weights_info and 'weight_max' in weights_info:
                new_layer = TrueQuantizedLinear(
                    module,
                    quantization_bits=weights_info.get('quantization_bits', 4),
                    reduced_rank=weights_info.get('reduced_rank', 16),
                    quantization_method=weights_info.get('quantization_method', 'uniform')
                )
                
                print(weights_info['qweight'])
                sys.exit()
                new_layer.register_buffer("qweight", weights_info['qweight'])
                new_layer.register_buffer("weight_max", weights_info['weight_max'])
                new_layer.register_buffer("weight_shape", weights_info['weight_shape'])
            else:
                print(f"Warning: Incomplete quantization info for {name}, cannot create TrueQuantLinearLayer")
                continue
        else:
            new_layer = LoraLinearLayer(
                module,
                quantization_bits=weights_info.get('quantization_bits', 4),
                reduced_rank=weights_info.get('reduced_rank', 16),
                quantization_method=weights_info.get('quantization_method', 'uniform')
            )
            
            # Set the dequantized parameters
            if 'dequantized_weight' in weights_info:
                new_layer.base_layer.weight.data = weights_info['dequantized_weight']
            
        if 'lora_A' in weights_info and 'lora_B' in weights_info:
            new_layer.lora_A.weight.data = weights_info['lora_A']
            new_layer.lora_B.weight.data = weights_info['lora_B']
            
        if new_layer.has_bias and 'bias' in weights_info:
            new_layer.bias.data = weights_info['bias']
                
        # Replace the module
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        
        if parent_name:
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, new_layer)
        else:
            setattr(model, child_name, new_layer)

    return model

def convert_true_quant_conv_layer(module: nn.Module, model_args):
    """
    Recursively iterates through a module and replaces nn.Conv2d layers
    with TrueQuantizedConv2d layers.
    """
    for child_name, child_module in module.named_children():
        if isinstance(child_module, nn.Conv2d):
            # Extract parameters from the original Conv2d layer
            in_channels = child_module.in_channels
            out_channels = child_module.out_channels
            kernel_size = child_module.kernel_size
            stride = child_module.stride
            padding = child_module.padding
            dilation = child_module.dilation
            groups = child_module.groups
            bias_exists = child_module.bias is not None

            # Create the TrueQuantizedConv2d layer
            # Ensure TrueQuantizedConv2d is imported or defined
            from loftq_cnn import TrueQuantizedConv2d # Adjust import path as needed

            new_layer = TrueQuantizedConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias_exists,
                quantization_bits=model_args.int_bit, # from your ModelArguments
                reduced_rank=model_args.reduced_rank, # from your ModelArguments
                num_iters=model_args.num_iter,       # from your ModelArguments
                quantization_method=model_args.quant_method # from your ModelArguments
                # Add other TrueQuantizedConv2d specific args if any from model_args
            )

            # Quantize the new layer using the weights of the original layer
            new_layer.quantize(child_module)

            # Replace the original layer with the new quantized layer
            setattr(module, child_name, new_layer)
            print(f"Replaced {child_name} with TrueQuantizedConv2d")

        elif len(list(child_module.children())) > 0:
            # Recursively apply to children
            convert_true_quant_conv_layer(child_module, model_args)
    return module

def analyze_model_memory(model):
    total_params = 0
    quant_params = 0
    quant_bytes = 0
    lora_params = 0
    other_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, TrueQuantizedLinear):
            # Quantized layer
            if hasattr(module, 'qweight'):
                quant_params += module.in_features * module.out_features
                quant_bytes += module.qweight.numel() * module.qweight.element_size()
            # LoRA adapters
            lora_params += module.lora_A.weight.numel() + module.lora_B.weight.numel()
        elif isinstance(module, LoraLinearLayer):
            if hasattr(module, 'base_layer'):
                param_count = module.base_layer.weight.numel()
                total_params += param_count
                other_params += param_count
            # LoRA adapters
            lora_params += module.lora_A.weight.numel() + module.lora_B.weight.numel()
        elif isinstance(module, nn.Linear):
            # Regular linear layer
            if hasattr(module, 'weight'):
                param_count = module.weight.numel()
                total_params += param_count
                other_params += param_count
    
    print(f"Quantized parameters: {quant_params:,} ({quant_bytes/(1024**2):.2f}MB)")
    print(f"LoRA parameters: {lora_params:,} ({lora_params*2/(1024**2):.2f}MB)")
    print(f"Other parameters: {other_params:,} ({other_params*2/(1024**2):.2f}MB)")
    print(f"Total: {other_params + quant_params + lora_params:,} ({((other_params*2) + quant_bytes + (lora_params*2))/(1024**2):.2f}MB)")