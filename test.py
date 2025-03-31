import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "LoftQ/Mistral-7B-v0.1-4bit-64rank"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Define a test prompt
prompt = "What are you?"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate response
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100)

# Decode and print response
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
