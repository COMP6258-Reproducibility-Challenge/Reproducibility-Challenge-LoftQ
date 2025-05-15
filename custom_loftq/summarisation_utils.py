from functools import partial
import logging
import math
import os
from filelock import FileLock

import numpy as np

import nltk
import evaluate
from transformers import (
    MBartTokenizer, 
    MBartTokenizerFast,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer
)
from transformers.utils import is_offline_mode
from datasets import DatasetDict

from utils import LoFTQTrainer, get_trained_save_dir

MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    "multi_news": ("document", "summary"),
}

def load_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except (LookupError, OSError):
        if is_offline_mode():
            raise LookupError(
                "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
            )
        with FileLock(".lock") as lock:
            nltk.download("punkt", quiet=True)
            
def resize_token_embeddings(model, tokenizer):
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    return model

def set_start_token(model, tokenizer, data_args):
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    
    return model

def max_positional_embeddings(model, model_args, data_args):
    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logging.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )
    
    return model

def prepare_model(model, tokenizer, data_args, model_args):
    model = resize_token_embeddings(model, tokenizer)
    model = set_start_token(model, tokenizer, data_args)
    model = max_positional_embeddings(model, model_args, data_args)
    return model

def compute_metrics_summarization(eval_preds, tokenizer, metric):
    preds, labels = eval_preds
    # if your model returns (logits, _), grab the logits
    if isinstance(preds, tuple):
        preds = preds[0]

    # replace -100 in labels (padding) so we can decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # decode predictions and references
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # simple post-processing: strip whitespace
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [lbl.strip() for lbl in decoded_labels]

    # compute ROUGE (returns floats for each rouge type)
    rouge_out = metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        rouge_types=["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    # round & convert to percentage
    result = {
        rouge_type: round(rouge_out[rouge_type] * 100, 4)
        for rouge_type in ["rouge1", "rouge2", "rougeL"]
    }

    # average generation length (in tokens)
    gen_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id)
        for pred in preds
    ]
    result["gen_len"] = float(np.mean(gen_lens))

    return result


def create_reduced_dataset(dataset, num_examples=25000):
    """Create a smaller subset of the dataset for quick testing."""
    
    # Sample dataset if too long
    max_len = max(len(split) for split in dataset.values())
    if max_len > num_examples:  # max_len > 40000:
        logging.warning(f"Raw dataset is too long and has been truncated to {num_examples} samples")
        percent = num_examples / max_len
        sampled_data = {}
        for split in dataset:
            sample_size = math.ceil(len(dataset[split]) * percent)
            sampled_data[split] = dataset[split].shuffle().select(range(sample_size))
        raw_data = DatasetDict(sampled_data)
        # print(f"Sampled dataset: {raw_data}")

    return raw_data


def preprocess_function(examples, tokenizer, prefix, text_column, summary_column, data_args, max_target_length, padding):
    inputs, targets = [], []
    for i in range(len(examples[text_column])):
        if examples[text_column][i] and examples[summary_column][i]:
            inputs.append(examples[text_column][i])
            targets.append(examples[summary_column][i])

    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

def train(model, tokenizer, model_args, data_args, training_args, raw_datasets):
    logging.warning("Preparing to train model")
    
    load_nltk()
    
    model = prepare_model(model, tokenizer, data_args, model_args)
    
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    
    column_names = raw_datasets["train"].column_names

    dataset_columns = summarization_name_mapping.get(data_args.data_name, None)
    
    text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    preprocess_with_tokenizer = partial(preprocess_function, tokenizer=tokenizer, prefix=prefix, text_column=text_column,
                                        summary_column=summary_column, data_args=data_args, max_target_length=max_target_length, padding=padding)
    
    metric = evaluate.load("rouge")
    compute_metrics = partial(compute_metrics_summarization, metric=metric, tokenizer=tokenizer)

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]
    if training_args.train_small:
        max_train_samples = min(len(train_dataset), 10)
        train_dataset = train_dataset.select(range(max_train_samples))
        max_eval_samples = min(len(eval_dataset), 10)
        eval_dataset = eval_dataset.select(range(max_eval_samples))
    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            preprocess_with_tokenizer,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on train dataset"
        )
    with training_args.main_process_first(desc="validation dataset map pre-processing"):
        eval_dataset = eval_dataset.map(
            preprocess_with_tokenizer,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on validation dataset",
        )
        
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )

    trainer = LoFTQSequence2SequenceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("Training model...")
    if (not training_args.resume_from_checkpoint is None):
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        train_result = trainer.train()
        
    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))
    
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Step 10: Evaluate on validation sets
    print("Evaluating model...")
    results = {}
    if isinstance(eval_dataset, dict):
        metrics = {}
        for eval_ds_name, eval_ds in eval_dataset.items():
            dataset_metrics = trainer.evaluate(eval_dataset=eval_ds, metric_key_prefix=f"eval_{eval_ds_name}")
            metrics.update(dataset_metrics)
    else:
        metrics = trainer.evaluate(metric_key_prefix="eval")
    max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Step 11: Save the fine-tuned model
    print("Saving final fine-tuned model...")
    model_dir = get_trained_save_dir(model_args, data_args.data_name, data_args.task_name)
    trainer.save_model(model_dir)

    print("Process completed!")
    

class LoFTQSequence2SequenceTrainer(Seq2SeqTrainer, LoFTQTrainer):
    def save_model(self, output_dir=None, _internal_call=False):
        LoFTQTrainer.save_model(self, output_dir, _internal_call)
    
    def _save(self, output_dir=None, state_dict=None):
        LoFTQTrainer._save(self, output_dir, state_dict)