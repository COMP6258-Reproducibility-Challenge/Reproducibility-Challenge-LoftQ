from functools import partial
import logging
import math

import evaluate
import torch.cuda
import transformers
from datasets import Dataset
from torch.nn import DataParallel

from model_utils import get_model_dir
from utils import LoFTQTrainer



def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds, metric):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return metric.compute(predictions=preds, references=labels)


def preprocessor(example, col_name, tokenizer):
    # result = {
    #     "input_ids": [],
    #     "attention_mask": [],
    #     "labels": [],
    # }
    #
    # for x in example:
    #     processed = tokenizer(x[col_name], padding="max_length", truncation=True, max_length=256)
    #     result["input_ids"].append(processed["input_ids"])
    #     result["attention_mask"].append(processed["attention_mask"])
    #     result["labels"].append(processed["input_ids"].copy())
    # print(tokenizer.pad_token)
    result = tokenizer(example[col_name], padding="max_length", truncation=True, max_length=256)
    result["labels"] = result["input_ids"].copy()
    return result

    # tokenizer.pad_token = tokenizer.eos_token
    # result = {
    #     "input_ids": [],
    #     "attention_mask": [],
    #     "labels": [],
    # }
    #
    # def batch(iterable):
    #     # This turns a list into a list of sub-lists of size batch_size
    #     l = len(iterable)
    #     for ndx in range(0, l, batch_size):
    #         yield iterable[ndx:min(ndx + batch_size, l)]
    #
    # for x in example[col_name]:
    #     processed = tokenizer(x, padding="max_length", truncation=True, max_length=256)
    #     result["input_ids"].append(processed["input_ids"])
    #     result["attention_mask"].append(processed["attention_mask"])
    #     result["labels"].append(processed["input_ids"].copy())
    #
    # return result



def train(model, tokenizer, model_args, training_args, raw_datasets):
    # Initialize
    print(transformers.__version__)
    logging.warning("Preparing to train model")
    metric = evaluate.load("accuracy")
    tokenizer.pad_token = tokenizer.eos_token
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    # Prepare datasets
    column_names = list(raw_datasets["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]
    processed_data = raw_datasets.map(
        lambda x: preprocessor(x, text_column_name, tokenizer),
        batched=True,
        remove_columns=column_names,
    )
    train_data = processed_data["train"]
    eval_data = processed_data["validation"]
    # train_data = IndexableDataLoader(raw_datasets, batch_size=8, collate_fn=lambda x: preprocessor(x, tokenizer))
    # train_data = raw_datasets["train"]
    # eval_data = raw_datasets["validation"]


    # Set up automatic trainer object
    trainer = LoFTQTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        compute_metrics=lambda pred: compute_metrics(pred, metric),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    print("Training model...")
    if (not training_args.resume_from_checkpoint is None):
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()

    # Step 10: Evaluate on validation sets
    print("Evaluating model...")
    metric_result = trainer.evaluate()
    metric_result["eval_samples"] = len(eval_data)
    try:
        perplexity = math.exp(metric_result["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metric_result["perplexity"] = perplexity



    # Step 11: Save the fine-tuned model
    print("Saving final fine-tuned model...")
    model_dir = get_model_dir("trained_models", model_args)
    trainer.save_model(model_dir)

    print("Process completed!")
