from typing import List, Optional, Set, Tuple, Type, Union

import os
import logging

from transformers import Trainer
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset

from model_utils import save_quantized_model

def load_raw_dataset(dataset_name: str, task_name: str) -> Tuple[Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset], Set, int]:
    logging.warning("Loading raw dataset")
    raw_data = load_dataset(dataset_name, task_name)
    return raw_data

class LoFTQTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        """Override save_model to use custom save function"""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Use custom save function
        save_quantized_model(self.model, self.processing_class, output_dir)
    
    def _save(self, output_dir=None, state_dict=None):
        """Override _save method as well"""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Use custom save function
        save_quantized_model(self.model, output_dir)