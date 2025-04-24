import copy
import logging
import os

import sys
from typing import Dict, Optional, Sequence

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset

import transformers
from peft import PeftModel, LoftQConfig, LoraConfig, TaskType, get_peft_model, LoraConfig, get_peft_model
from dataclasses import dataclass, field
from datasets import load_dataset

import utils


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
ANSWER_PROMPT = "The final answer is: "
QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"

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
        default="glue",
        metadata={"help": "Dataset name."}
    )
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    expt_name: str = field(
        default="default",
        metadata={"help": "Experiment name"},
    )
    
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def preprocess(sources: Sequence[str], targets: Sequence[str], labels: Sequence[int], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Preprocess the data by tokenizing."""
    inputs = tokenizer(
        sources,
        text_pair=targets,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
    )
    
    inputs["labels"] = torch.tensor(labels)
    
    return inputs
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Formatting inputs...")
        sources, targets, labels = tuple([d[k] for d in raw_data] for k in ["premise", "hypothesis", "label"])
    
        # Get unique labels instead of all labels
        self.label_list = sorted(list(set(labels)))
        self.num_labels = len(self.label_list)
        
        # Create a mapping from label values to indices
        self.label_to_id = {label: i for i, label in enumerate(self.label_list)}
        
        # Convert string/text labels to indices if needed
        numeric_labels = [self.label_to_id[label] for label in labels]

        logging.warning("Tokenizing inputs... This may take some time...")
        self.inputs = preprocess(sources, targets, numeric_labels, tokenizer)
        print(f"Number of unique labels: {self.num_labels}")

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return {k: v[i] for k, v in self.inputs.items()}
    
    def get_labels(self):
        return self.label_list
    
    def get_num_labels(self):
        return self.num_labels


# @dataclass
# class DataCollatorForSupervisedDataset(object):
#     """Collate examples for supervised fine-tuning."""

#     tokenizer: transformers.PreTrainedTokenizer

#     def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
#         input_ids = torch.nn.utils.rnn.pad_sequence(
#             input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
#         )
#         labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
#         return dict(
#             input_ids=input_ids,
#             labels=labels,
#             attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
#         )

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Create a batch by stacking the individual instances
        batch = {}
        
        # For classification tasks, labels are already integer indices
        for key in instances[0].keys():
            if key == "labels":
                # For classification, labels should already be tensors of shape (batch_size,)
                batch[key] = torch.stack([instance[key] for instance in instances])
            elif key == "input_ids" or key == "attention_mask":
                # These might need padding if not already padded
                if isinstance(instances[0][key], torch.Tensor) and instances[0][key].dim() == 1:
                    # If they're 1D tensors, pad them
                    batch[key] = torch.nn.utils.rnn.pad_sequence(
                        [instance[key] for instance in instances], 
                        batch_first=True, 
                        padding_value=self.tokenizer.pad_token_id if key == "input_ids" else 0
                    )
                else:
                    # If they're already 2D tensors (pre-padded), just stack them
                    batch[key] = torch.stack([instance[key] for instance in instances])
            else:
                # For any other keys
                values = [instance[key] for instance in instances]
                if isinstance(values[0], torch.Tensor):
                    batch[key] = torch.stack(values)
                else:
                    batch[key] = values
                    
        # Ensure attention mask is set correctly
        if "attention_mask" not in batch:
            batch["attention_mask"] = batch["input_ids"].ne(self.tokenizer.pad_token_id)
            
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    logging.warning("Downloading Data")
    dataset = load_dataset(data_args.data_name, "mnli")
    train_set = dataset['train']
    train_dataset = SupervisedDataset(raw_data=train_set, tokenizer=tokenizer)
    # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_collator = transformers.DataCollatorWithPadding(tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

    
def load_model(training_args):
    model_path = "./quantized_model"
    config = transformers.AutoConfig.from_pretrained(model_path)
    model = transformers.AutoModelForSequenceClassification.from_config(config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    tokenizer.model_max_length=training_args.model_max_length
    
    peft_config = LoraConfig.from_pretrained("./peft_loftq_model")
    
    peft_model = get_peft_model(model, peft_config)
    
    # Print trainable parameters info
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    print(f"Trainable parameters: {trainable_params} ({trainable_params/total_params:.2%} of total)")
    
    return peft_model, tokenizer
    
def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    model, tokenizer = load_model(training_args)
    
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    model.config.num_labels = data_module['train_dataset'].get_num_labels()
    training_args.output_dir = os.path.join(
        training_args.output_dir,
        training_args.expt_name,
        model_args.model_name_or_path.split('/')[-1],
        f"ep_{int(training_args.num_train_epochs)}",
        f"lr_{training_args.learning_rate}",
        f"seed_{training_args.seed}",
    )
    trainer = transformers.Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    
    
if __name__ == "__main__":
    train()