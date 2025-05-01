# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import sys # Added for flushing output
import time # Added for timing

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from peft import LoftQConfig, LoraConfig, TaskType, get_peft_model
from safetensors import safe_open


class Shell(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)


def unwrap_model(model, sub_module_name=".base_layer"):
    """
    Replaces LoRA layers with their base linear layers (Shell).
    """
    print(f"\n===== Starting Model Unwrapping (Submodule: '{sub_module_name}') =====")
    start_time = time.time()
    sub_module_name_list = [k.split(sub_module_name)[0] for k in model.state_dict().keys() if sub_module_name in k]
    sub_module_name_set = set(sub_module_name_list)
    print(f"Found {len(sub_module_name_set)} modules to unwrap.")

    unwrapped_count = 0
    for name in sub_module_name_set:
        try:
            # get the parent of the submodule
            name_parent = ".".join(name.split(".")[:-1])
            name_child = name.split(".")[-1]
            sub_module = model.get_submodule(name_parent)
            # print(f"  Processing module: {name}")
            # print(f"    Parent: {name_parent}, Child: {name_child}")
            # print(f"    Submodule type: {type(sub_module)}")

            # replace with shell
            child = getattr(sub_module, name_child)
            # print(f"    Child module type: {type(child)}")

            if hasattr(child, 'base_layer'):
                weight = getattr(child.base_layer, "weight", None)
                bias = getattr(child.base_layer, "bias", None)
                if weight is not None:
                    print(f"  Unwrapping: {name} (Replacing {type(child).__name__} with Shell)")
                    shell = Shell(weight, bias)
                    setattr(sub_module, name_child, shell)
                    unwrapped_count += 1
                else:
                    print(f"  Skipping {name}: No 'weight' found in base_layer.")
            else:
                 print(f"  Skipping {name}: No 'base_layer' attribute found in child module {type(child).__name__}.")

        except Exception as e:
            print(f"  Error unwrapping {name}: {e}")

    end_time = time.time()
    print(f"Successfully unwrapped {unwrapped_count} modules.")
    print(f"Model unwrapping finished in {end_time - start_time:.2f} seconds.")
    print("===== Finished Model Unwrapping =====")
    # Original print statement:
    # print("You have unwrapped the model. Use it on your own risk.")


def print_model(model, name):
    """Prints model summary and parameter details."""
    print("\n" + "=" * 15 + f" Model Summary: {name} " + "=" * 15)
    # print(model) # This can be very long, uncomment if needed
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {num_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    if num_params > 0:
      print(f"Trainable Percentage: {100 * trainable_params / num_params:.2f}%")

    # print("\n--- Parameter Details ---")
    # for param_name, param in model.named_parameters():
    #     if torch.is_tensor(param):
    #         details = [
    #             f"Shape: {tuple(param.shape)}",
    #             f"Device: {param.device}",
    #             f"Dtype: {param.dtype}",
    #             f"Grad: {param.requires_grad}"
    #         ]
    #         if param.dtype in [torch.float32, torch.float16, torch.bfloat16] and param.numel() > 0:
    #             try:
    #                 details.append(f"Mean: {param.mean().item():.4f}")
    #                 details.append(f"Max: {param.max().item():.4f}")
    #             except Exception: # Handle potential issues with empty tensors or weird states
    #                 details.append("Stats N/A")
    #         print(f"  {param_name:<50} | " + " | ".join(details))
    #     else:
    #         print(f"  {param_name:<50} | Not a tensor")
    print("=" * (30 + len(name) + 2))


def arg_parse():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Quantize a model with LoftQ.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="The name or path of the fp32/16 model.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="The access token to download model from HuggingFace Hub.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="The quantized bits",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=1,
        help="The alternating steps in LoftQ",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="The rank of the LoRA adapter",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./model_zoo/loftq/",
        help="Directory to save the quantized model and adapters.",
    )
    print("Parsing command-line arguments...")
    args = parser.parse_args()
    print("Arguments parsed:")
    for arg, value in vars(args).items():
        print(f"  --{arg}: {value}")
    print("-" * 30)
    return args


def quantize_and_save():
    """Main function to quantize the model and save components."""
    args = arg_parse()
    start_total_time = time.time()

    # --- 1. Load Tokenizer ---
    print(f"\n[1/7] Loading tokenizer for '{args.model_name_or_path}'...")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=args.token, trust_remote_code=True)
    end_time = time.time()
    print(f"Tokenizer loaded in {end_time - start_time:.2f} seconds.")
    sys.stdout.flush()

    # --- 2. Load Model & Determine Task Type ---
    print(f"\n[2/7] Loading model '{args.model_name_or_path}'...")
    print(f"      (Using token: {'Yes' if args.token else 'No'})")
    start_time = time.time()
    model_type_detected = "Unknown"
    if any(name in args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "phi"]): # Added phi
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16, # Using bfloat16 as recommended
            token=args.token,
            trust_remote_code=True,
            device_map={"": torch.cuda.current_device()} if torch.cuda.is_available() else None, # Ensure mapping to current GPU
            # low_cpu_mem_usage=True # Can be helpful for large models if RAM is limited
        )
        # model.to('cuda') # Explicitly move model to CUDA if device_map wasn't used or failed
        task_type = TaskType.CAUSAL_LM
        if 'phi' in args.model_name_or_path.lower():
             target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"] # Phi specific
        else:
             target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        model_type_detected = "Causal LM"

    elif any(name in args.model_name_or_path.lower() for name in ["bart", "t5"]):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            token=args.token,
            device_map="auto" # Use auto device map
            )
        task_type = TaskType.SEQ_2_SEQ_LM
        target_modules = ["q_proj", "k_proj", "v_proj", "fc1", "fc2", "out_proj"]
        model_type_detected = "Seq2Seq LM"

    elif any(name in args.model_name_or_path.lower() for name in ["deberta", "roberta", "bert"]):
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            token=args.token,
            device_map="auto" # Use auto device map
            )
        task_type = TaskType.SEQ_CLS
        target_modules = ["query_proj", "key_proj", "value_proj", "dense"]  # embeddings not supported by peft
        model_type_detected = "Sequence Classification"
    else:
        raise NotImplementedError(f"Model type for '{args.model_name_or_path}' not recognized or supported yet.")

    # Ensure model is on GPU if CUDA is available
    if torch.cuda.is_available() and next(model.parameters()).device.type != 'cuda':
         print("Model not on GPU, moving to CUDA...")
         model.to(torch.device("cuda"))

    end_time = time.time()
    print(f"Model loaded ({model_type_detected}) in {end_time - start_time:.2f} seconds.")
    print(f"Target modules for LoRA: {target_modules}")
    sys.stdout.flush()

    # --- 3. Configure LoftQ and LoRA ---
    print(f"\n[3/7] Configuring LoftQ (bits={args.bits}, iter={args.iter}) and LoRA (rank={args.rank})...")
    start_time = time.time()
    loftq_config = LoftQConfig(loftq_bits=args.bits, loftq_iter=args.iter)

    # Determine lora_alpha based on recommendations
    lora_alpha = args.rank
    if task_type is TaskType.CAUSAL_LM and args.bits == 4:
        lora_alpha = 16 # Recommended default for 4-bit LMs
        print(f"  (Setting lora_alpha to {lora_alpha} based on task type and bits)")


    lora_config = LoraConfig(
        task_type=task_type,
        inference_mode=True, # Important: Set to True during quantization phase
        r=args.rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.1, # Dropout is typically not used during inference/quantization
        target_modules=target_modules,
        init_lora_weights="loftq", # Specify LoftQ initialization
        loftq_config=loftq_config,
    )
    end_time = time.time()
    print(f"LoftQ and LoRA configured in {end_time - start_time:.2f} seconds.")
    print("LoftQ Config:", loftq_config)
    print("LoRA Config:", lora_config)
    sys.stdout.flush()

    # --- 4. Apply PEFT to Get LoftQ Model ---
    print(f"\n[4/7] Applying PEFT with LoftQ configuration (This is the quantization step)...")
    start_time = time.time()
    lora_model = get_peft_model(model, lora_config)
    end_time = time.time()
    print(f"PEFT model with LoftQ obtained in {end_time - start_time:.2f} seconds.")
    print_model(lora_model, "Initial PEFT (LoRA) Model")
    sys.stdout.flush()

    # --- 5. Save LoRA Adapters ---
    print(f"\n[5/7] Saving LoRA model components...")
    start_time = time.time()
    # Define directories
    model_name_suffix = f"-{args.bits}bit-{args.rank}rank"
    model_base_name = args.model_name_or_path.split("/")[-1]
    model_name = model_base_name + model_name_suffix
    base_model_dir = os.path.join(args.save_dir, model_name)
    lora_adapter_dir = os.path.join(base_model_dir, "loftq_init") # Save adapters in subfolder
    os.makedirs(lora_adapter_dir, exist_ok=True) # Ensure adapter dir exists

    print(f"  Saving LoRA adapters to: {lora_adapter_dir}")
    lora_model.save_pretrained(lora_adapter_dir) # Saves adapter_model.bin/safetensors and adapter_config.json

    # Convert safetensor to bin if it exists (optional, for compatibility)
    safetensors_path = os.path.join(lora_adapter_dir, "adapter_model.safetensors")
    bin_path = os.path.join(lora_adapter_dir, "adapter_model.bin")
    if os.path.exists(safetensors_path):
        print(f"  Converting '{safetensors_path}' to '{bin_path}'...")
        tensors = {}
        try:
            with safe_open(safetensors_path, framework="pt") as f:
                for k in f.keys():
                    tensors[k] = f.get_tensor(k)
            torch.save(tensors, bin_path)
            print("  Conversion successful.")
        except Exception as e:
            print(f"  Warning: Failed to convert safetensors to bin: {e}")
    else:
        print("  No adapter_model.safetensors found, skipping conversion.")


    # Modify adapter_config.json for future loading
    adapter_config_path = os.path.join(lora_adapter_dir, "adapter_config.json")
    print(f"  Modifying adapter configuration file: {adapter_config_path}")
    try:
        with open(adapter_config_path, "r") as fp:
            adapter_config = json.load(fp)

        adapter_config['base_model_name_or_path'] = base_model_dir # Point to the saved base model directory
        adapter_config['init_lora_weights'] = True # Crucial: Prevent re-applying LoftQ on load
        adapter_config['inference_mode'] = True # Should be loaded in inference mode later

        with open(adapter_config_path, "w") as fp:
            json.dump(adapter_config, fp, indent=2)
        print("  Adapter configuration updated successfully.")
    except Exception as e:
        print(f"  Error modifying adapter_config.json: {e}")


    end_time = time.time()
    print(f"LoRA adapters saved and configured in {end_time - start_time:.2f} seconds.")
    sys.stdout.flush()


    # --- 6. Prepare and Save Base Model ---
    print(f"\n[6/7] Preparing and saving the quantized base model...")
    start_time = time.time()
    print(f"  Getting base model from PEFT model...")
    base_model = lora_model.get_base_model()
    print(f"  Unwrapping base model (replacing LoRA layers with Shell)...")
    unwrap_model(base_model) # Replace LoRA layers with shells containing quantized weights

    print(f"  Saving unwrapped base model to: {base_model_dir}")
    os.makedirs(base_model_dir, exist_ok=True) # Ensure base model dir exists
    base_model.save_pretrained(base_model_dir) 
    print(f"  Saving tokenizer to: {base_model_dir}")
    tokenizer.save_pretrained(base_model_dir)
    end_time = time.time()
    print_model(base_model, "Final Base Model (Quantized, Unwrapped)")
    print(f"Base model and tokenizer saved in {end_time - start_time:.2f} seconds.")
    sys.stdout.flush()

    # --- 7. Completion ---
    print(f"\n[7/7] LoftQ quantization and saving process completed.")
    print(f"  Quantized Base Model saved to: {base_model_dir}")
    print(f"  LoRA Initial Adapters saved to: {lora_adapter_dir}")
    end_total_time = time.time()
    print(f"\nTotal execution time: {end_total_time - start_total_time:.2f} seconds.")

    return base_model_dir, lora_adapter_dir


if __name__ == "__main__":
    print("Starting LoftQ Quantization Script...")
    try:
        base_dir, lora_dir = quantize_and_save()
    except Exception as e:
        print(f"\n!!! An error occurred: {e} !!!")
        import traceback
        traceback.print_exc()
    finally:
        print("\nScript finished.")


# example command:
# python quantize_save.py \
# --model_name_or_path meta-llama/Llama-2-7b-hf \
# --token XXX \
# --bits 4 --iter 1 --rank 16 \
# --save_dir ./model_zoo/loftq/