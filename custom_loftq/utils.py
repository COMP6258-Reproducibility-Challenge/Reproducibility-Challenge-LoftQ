from typing import List, Optional, Set, Tuple, Type, Union

import os
import logging

from transformers import Trainer
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset

from model_utils import save_quantized_model, get_model_dir

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


def index_dim(tensor, dim, start, block_size):
    """
    Indexes the tensor along a specified dimension with a specified starting element and ending element
    Args:
        tensor: tensor to index
        dim: specified dimension
        start: start element index within the dimension
        block_size: end element index within the dimension

    Returns:
        a slice of the tensor along dimension dim
    """
    # Create slice objects for all dimensions
    slices = [slice(None)] * tensor.dim()

    # Replace the slice for dim with the desired range
    slices[dim] = slice(start, start + block_size)

    # Index the tensor
    return tensor[tuple(slices)]


def max_dim(m):
    """
    Finds the dimension with the most amount of elements in a tensor and the number of elements
    inside that dimension inside a tuple.
    Args:
        m: input tensor

    Returns:
        Tuple of the form (dim, numel)
    """
    if m.dim() <= 1:
        return -m.dim(), max(m.shape) if m.dim() == 1 else 0
    return min(range(-m.dim(), -1), key=lambda d: -(m.shape[d] + 1)), max(m.shape)

def get_trained_save_dir(model_args, data_name, task_name):
    save_dir = "trained_models"
    save_dir = os.path.join(save_dir, data_name)
    if task_name != "main":
        save_dir = os.path.join(save_dir, task_name)
    return get_model_dir(save_dir, model_args)