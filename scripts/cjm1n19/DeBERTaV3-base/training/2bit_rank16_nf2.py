import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_DIR = "~/LoftQ/models/deberta-v3-base-2bit-16rank"

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, 
    torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type='nf4',
    ),
)
peft_model = PeftModel.from_pretrained(
    base_model,
    MODEL_DIR,
    subfolder="loftq_init",
    is_trainable=True,
)
# Do training with peft_model ...
