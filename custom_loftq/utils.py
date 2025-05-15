from typing import List, Optional, Set, Tuple, Type, Union

import os
import logging

import math

import datasets
from transformers import Trainer
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset, concatenate_datasets
from peft import TaskType

from model_utils import save_quantized_model, get_model_dir

def load_raw_dataset(dataset_name: str, task_name: str) -> Tuple[Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset], Set, int]:
    logging.warning(f"Loading raw dataset: {dataset_name} - {task_name}")
    raw_data = load_dataset(dataset_name, task_name)

    if dataset_name == 'anli':
        train = concatenate_datasets([
            raw_data['train_r1'],
            raw_data['train_r2'],
            raw_data['train_r3']
        ])
        dev = concatenate_datasets([
            raw_data['dev_r1'],
            raw_data['dev_r2'],
            raw_data['dev_r3']
        ])
        test = concatenate_datasets([
            raw_data['test_r1'],
            raw_data['test_r2'],
            raw_data['test_r3']
        ])
        raw_data = DatasetDict({
            "train": train,
            "validation": dev,
            "test": test,
        })

    return raw_data

class LoFTQTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        """Override save_model to use custom save function"""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Use custom save function
        save_quantized_model(self.model, self.tokenizer, output_dir)
    
    def _save(self, output_dir=None, state_dict=None):
        """Override _save method as well"""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Use custom save function
        save_quantized_model(self.model, self.tokenizer, output_dir)

def get_trained_save_dir(model_args, data_name, task_name):
    save_dir = "trained_models"
    save_dir = os.path.join(save_dir, data_name)
    if task_name != "main":
        save_dir = os.path.join(save_dir, task_name)
    return get_model_dir(save_dir, model_args)