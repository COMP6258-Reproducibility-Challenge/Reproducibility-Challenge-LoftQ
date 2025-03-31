import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# fetch the MODEL_ID at https://huggingface.co/LoftQ
MODEL_ID = "LoftQ/Mistral-7B-v0.1-4bit-64rank"

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16,  # you may change it with different models
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 is recommended
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type='nf4',
    ),
)
peft_model = PeftModel.from_pretrained(
    base_model,
    MODEL_ID,
    subfolder="loftq_init",
    is_trainable=True,
)

# Do training with peft_model ...