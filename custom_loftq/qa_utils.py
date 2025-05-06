from functools import partial
import logging

import numpy as np

import evaluate

from model_utils import get_model_dir
from utils import LoFTQTrainer

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def count_labels(dataset):
    train_set = dataset["train"]
    labels = set([entry['answer'] for entry in train_set])
    num_labels = len(labels)
    return num_labels, labels


def compute_classification_metrics(eval_pred, data_args):
    metric = evaluate.load(data_args.data_name, data_args.task_name)
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def create_reduced_dataset(dataset, labels, num_examples=1000, seed=42):
    """Create a smaller subset of the dataset for quick testing."""
    # Set seed for reproducibility
    import numpy as np
    np.random.seed(seed)

    # Get a balanced subset
    indices = []
    for label in labels:  # MNLI has 3 classes
        label_indices = [i for i, l in enumerate(dataset["label"]) if l == label]
        selected = np.random.choice(label_indices, min(num_examples // 3, len(label_indices)), replace=False)
        indices.extend(selected)

    # Shuffle the indices
    np.random.shuffle(indices)

    # Create the reduced dataset
    reduced_dataset = dataset.select(indices)
    print(f"Created reduced dataset with {len(reduced_dataset)} examples")

    return reduced_dataset


def preprocess_function(examples, tokenizer, sentence1_key, sentence2_key):
    return tokenizer(
        examples[sentence1_key],
        examples[sentence2_key],
        padding="max_length",
        truncation=True,
        max_length=256
    )


def train(model, tokenizer, model_args, data_args, training_args, raw_datasets):
    logging.warning("Preparing to train model")

    sentence1_key, sentence2_key = task_to_keys[data_args.task_name]

    preprocess_with_tokenizer = partial(preprocess_function, tokenizer=tokenizer, sentence1_key=sentence1_key,
                                        sentence2_key=sentence2_key)
    compute_metrics = partial(compute_classification_metrics, data_args=data_args)

    columns_to_remove = [col for col in raw_datasets['train'].column_names if col != "label"]

    tokenized_datasets = raw_datasets.map(
        preprocess_with_tokenizer,
        batched=True,
        remove_columns=columns_to_remove
    )

    trainer = LoFTQTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Training model...")
    if (not training_args.resume_from_checkpoint is None):
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()

    # Step 10: Evaluate on validation sets
    print("Evaluating model...")
    if data_args.task_name == "mnli":
        results_matched = trainer.evaluate(tokenized_datasets["validation_matched"])
        results_mismatched = trainer.evaluate(tokenized_datasets["validation_mismatched"])
        print(f"Validation Matched Results: {results_matched}")
        print(f"Validation Mismatched Results: {results_mismatched}")
    else:
        results = trainer.evaluate(tokenized_datasets["validation"])
        print(f"Validation Matched Results: {results}")

    # Step 11: Save the fine-tuned model
    print("Saving final fine-tuned model...")
    model_dir = get_model_dir("trained_models", model_args)
    trainer.save_model(model_dir)

    print("Process completed!")