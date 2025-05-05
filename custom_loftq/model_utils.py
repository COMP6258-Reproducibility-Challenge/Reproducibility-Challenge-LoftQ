"""model_utils_fixed.py – LoFTQ / AdaNF aware helpers

This is a drop‑in replacement for the original `model_utils.py` and keeps the
public API unchanged while fixing persistence of AdaNF per‑block offsets and
retaining LoRA adapters during load.
"""
import logging
import os
from typing import List, Optional, Tuple, Type, Union

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoConfig,
    PreTrainedModel,
)
from peft import TaskType

from loftq_adanf_fixed import (
    convert_linear_layer,
    LoraLinearLayer,
    TrueQuantizedLinear,
    BaseLoftqLinear,
)
from arguments import ModelArguments

# ---------------------------------------------------------------------------
# Utility bookkeeping
# ---------------------------------------------------------------------------

def count_trainable_parameters(model: Union[PreTrainedModel, torch.nn.Module]):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: Union[PreTrainedModel, torch.nn.Module]):
    return sum(p.numel() for p in model.parameters())


def get_model_dir(save_dir: str, model_args: ModelArguments) -> str:
    model_name = (
        model_args.model_name_or_path.split("/")[-1]
        + f"-{model_args.int_bit}bit"
        + f"-{model_args.reduced_rank}rank"
    )
    return os.path.join(save_dir, model_name)


# ---------------------------------------------------------------------------
# Backbone helpers (unchanged)
# ---------------------------------------------------------------------------

def get_base_class(model_name: str) -> Type[PreTrainedModel]:
    config = AutoConfig.from_pretrained(model_name)
    model_type = config.model_type

    if any(name in model_name.lower() for name in ["llama", "mistral", "falcon"]):
        return AutoModelForCausalLM
    elif any(name in model_name.lower() for name in ["bart", "t5"]):
        return AutoModelForSeq2SeqLM
    elif any(name in model_name.lower() for name in ["deberta", "roberta", "bert"]):
        if model_type == "bert":
            from transformers import BertForSequenceClassification as ModelClass
        elif model_type == "roberta":
            from transformers import RobertaForSequenceClassification as ModelClass
        else:
            from transformers import DebertaV2ForSequenceClassification as ModelClass
        return ModelClass
    else:
        raise NotImplementedError("Other models not supported yet.")


def get_target_excluded_modules(model_name: str) -> Tuple[List[str], List[str]]:
    if any(name in model_name.lower() for name in ["llama", "mistral", "falcon"]):
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ]
    elif any(name in model_name.lower() for name in ["bart", "t5"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "fc1", "fc2", "out_proj"]
    elif any(name in model_name.lower() for name in ["deberta", "roberta", "bert"]):
        target_modules = ["query_proj", "key_proj", "value_proj", "dense"]
    else:
        raise NotImplementedError("Other models not supported yet.")

    excluded_modules = ["classifier", "LayerNorm", "pooler"]
    return target_modules, excluded_modules


# ---------------------------------------------------------------------------
# Loading the base HF model
# ---------------------------------------------------------------------------

def load_base_model(
    model_name: str, token: str, num_labels: Optional[int]
) -> Tuple[Union[PreTrainedModel, torch.nn.Module], AutoTokenizer, List[str], List[str]]:
    logging.info("Loading base model …")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    if any(name in model_name.lower() for name in ["llama", "mistral", "falcon"]):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            token=token,
            trust_remote_code=True,
            device_map="auto",
        )
        task_type = TaskType.CAUSAL_LM  # noqa: F841 – kept for future use
    elif any(name in model_name.lower() for name in ["bart", "t5"]):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=token)
        task_type = TaskType.SEQ_2_SEQ_LM
    elif any(name in model_name.lower() for name in ["deberta", "roberta", "bert"]):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, token=token, num_labels=num_labels
        )
        task_type = TaskType.SEQ_CLS
    else:
        raise NotImplementedError("Other models not supported yet.")

    target_modules, excluded_modules = get_target_excluded_modules(model_name)
    return model, tokenizer, target_modules, excluded_modules


# ---------------------------------------------------------------------------
# Quantisation wrapper
# ---------------------------------------------------------------------------

def quantize_model(
    model: Union[PreTrainedModel, torch.nn.Module],
    model_args: ModelArguments,
    target_modules: List[str],
    excluded_modules: List[str],
):
    logging.info("Quantizing model – this may take a while …")
    model = convert_linear_layer(
        model,
        quantization_bits=model_args.int_bit,
        rank=model_args.reduced_rank,
        quantization_method=model_args.quant_method,
        target_modules=target_modules,
        excluded_modules=excluded_modules,
        true_quantization=model_args.true_quantization,
    )
    return model


# ---------------------------------------------------------------------------
# *** Saving *** (now AdaNF‑aware)
# ---------------------------------------------------------------------------

def save_quantized_model(
    model: Union[PreTrainedModel, torch.nn.Module],
    tokenizer: AutoTokenizer,
    save_path: str,
    model_args: Optional[ModelArguments] = None,
):
    """Save HF config + LoFTQ weights + mapping table, including AdaNF offsets."""

    os.makedirs(save_path, exist_ok=True)
    model_dir = get_model_dir(save_path, model_args) if model_args else save_path

    # 1. Standard HF save (best‑effort)
    if isinstance(model, PreTrainedModel):
        try:
            model.save_pretrained(model_dir)
        except RuntimeError as e:
            logging.warning(f"save_pretrained failed: {e}; falling back to custom save")
            if hasattr(model, "config"):
                model.config.save_pretrained(model_dir)
    else:
        torch.save(model.state_dict(), os.path.join(model_dir, "loftq_model.pt"))

    # 2. Collect LoFTQ‑specific state
    loftq_state_dict = {}
    module_mapping = {}

    for name, module in model.named_modules():
        if not issubclass(type(module), BaseLoftqLinear):
            continue

        mod_state = {
            "lora_A": module.lora_A.weight.data,
            "lora_B": module.lora_B.weight.data,
        }
        if module.has_bias and module.bias is not None:
            mod_state["bias"] = module.bias.data

        module_mapping[name] = {
            "type": "BaseLoftqLinear",
            "in_features": module.in_features,
            "out_features": module.out_features,
            "reduced_rank": module.reduced_rank,
            "quantization_bits": module.quantization_bits,
            "num_iters": module.num_iters,
            "quantization_method": module.quantization_method,
        }

        # ---- packed weight + AdaNF offsets (inner TrueQuantizedLinear) -----
        qlin = module.qlinear
        if isinstance(qlin, TrueQuantizedLinear):
            mod_state.update(
                {
                    "qweight": qlin.qweight,
                    "weight_max": qlin.weight_max,
                    "weight_shape": qlin.weight_shape,
                }
            )
            if getattr(qlin, "c_offset_idx", None) is not None:
                mod_state["c_offset_idx"] = qlin.c_offset_idx

        loftq_state_dict[name] = mod_state

    torch.save(loftq_state_dict, os.path.join(model_dir, "loftq_weights.pt"))
    torch.save(module_mapping, os.path.join(model_dir, "loftq_mapping.pt"))
    tokenizer.save_pretrained(model_dir)


# ---------------------------------------------------------------------------
# *** Loading *** (keeps wrapper, restores AdaNF offsets)
# ---------------------------------------------------------------------------

def load_loftq_model(
    model_class: Type[PreTrainedModel],
    model_args: ModelArguments,
    save_dir: str,
):
    """Load HF model + LoFTQ layers with all buffers including AdaNF offsets."""

    model_dir = get_model_dir(save_dir, model_args)

    # 1. Base HF model
    if issubclass(model_class, PreTrainedModel):
        model = model_class.from_pretrained(model_dir)
    else:
        model = model_class(**(model_args or {}))
        model.load_state_dict(torch.load(os.path.join(model_dir, "loftq_model.pt")), strict=False)

    module_mapping = torch.load(os.path.join(model_dir, "loftq_mapping.pt"))

    # 2. Ensure wrappers exist (they should after save) – but don’t replace them
    for name, info in module_mapping.items():
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        wrapper = parent.get_submodule(child_name)

        if not issubclass(type(wrapper), BaseLoftqLinear):
            logging.warning(f"Missing LoFTQ wrapper at {name}; reconstructing …")
            base = wrapper if isinstance(wrapper, torch.nn.Linear) else torch.nn.Linear(
                info["in_features"], info["out_features"], bias=True
            )
            wrapper = convert_linear_layer(
                base,
                quantization_bits=info["quantization_bits"],
                rank=info["reduced_rank"],
                quantization_method=info["quantization_method"],
                true_quantization=True,
            )
            setattr(parent, child_name, wrapper)

    # 3. Load packed weights + LoRA params
    loftq_state_dict = torch.load(os.path.join(model_dir, "loftq_weights.pt"))
    with torch.no_grad():
        for name, params in loftq_state_dict.items():
            wrapper = model.get_submodule(name)
            qlin = wrapper.qlinear
            for pname, tensor in params.items():
                if pname == "lora_A":
                    wrapper.lora_A.weight.copy_(tensor.t())
                elif pname == "lora_B":
                    wrapper.lora_B.weight.copy_(tensor)
                elif pname == "bias" and wrapper.bias is not None:
                    wrapper.bias.copy_(tensor)
                elif pname in {"qweight", "weight_max", "weight_shape", "c_offset_idx"}:
                    qlin.register_buffer(pname, tensor)

    # 4. Return extras needed by trainer
    target_modules, excluded_modules = get_target_excluded_modules(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer, target_modules, excluded_modules
