import transformers
import datasets
import torch

def preprocessor(example, col_name, tokenizer):
    result = tokenizer(example[col_name], padding="max_length", truncation=True, max_length=256)
    result["labels"] = result["input_ids"].copy()
    return result

model = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="auto")
tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token


ds = datasets.load_dataset("mikasenghaas/wikitext-2", "default")
column_names = list(ds["train"].features)
text_column_name = "text" if "text" in column_names else column_names[0]
ds = ds.map(
    lambda x: preprocessor(x, text_column_name, tokenizer),
    batched=True,
    remove_columns=column_names,
)

print(f"Model is on cuda: {next(model.parameters()).is_cuda}")
model(
    input_ids=torch.tensor([ds["train"][0]["input_ids"]]).to("cuda"),
    attention_mask=torch.tensor([ds["train"][0]["attention_mask"]]).to("cuda"),
    labels=torch.tensor([ds["train"][0]["labels"]]).to("cuda"),
)
model.zero_grad()
print("success")

