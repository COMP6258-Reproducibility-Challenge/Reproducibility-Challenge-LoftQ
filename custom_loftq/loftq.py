import math
import sys
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn
from accelerate.utils.memory import clear_device_cache
import logging


compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class LoraLinearLayer(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        quantization_bits: int = 4,
        reduced_rank: int = 16,
        num_iters: int = 5,
        quantization_method: str = "uniform"
    ):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        self.base_layer = base_layer

        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        self.quantization_bits = quantization_bits
        self.reduced_rank = reduced_rank
        self.num_iters = num_iters
        self.quantization_method = quantization_method
        
        self.lora_A = nn.Linear(self.in_features, reduced_rank, bias=False)
        self.lora_B = nn.Linear(reduced_rank, self.out_features, bias=False)
        
        self.reset_lora_parameters()
        
        self.has_bias = base_layer.bias is not None
        if self.has_bias:
            self.bias = nn.Parameter(base_layer.bias.data.clone())
        else:
            self.register_parameter('bias', None)
        
    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def quantize(self):
        W_initial = self.base_layer.weight.data.clone()
        
        Q, A, B = quantize_layer(W_initial, self.quantization_bits, self.reduced_rank, self.num_iters, self.quantization_method)
        
        self.base_layer.weight.data = Q
        self.lora_A.weight.data = A
        self.lora_B.weight.data = B
    
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        out = self.base_layer(x, *args, **kwargs)
        # out = nn.functional.linear(x, self.weight, self.bias)
        lora = self.lora_B(self.lora_A(x))
        return out + lora


class BlockQuantizer:
    def __init__(self, num_bits=2, method="uniform", block_size=64, device="cpu", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_bits = num_bits
        self.device = device
        self.method = method
        self.block_size = block_size
        self.method = method
        if self.method == "normal":
            self.norm_lookup_table = self.create_normal_map(num_bits=self.num_bits)
            self.norm_lookup_table = self.norm_lookup_table.to(device)
        elif self.method == "uniform":
            self.norm_lookup_table = self.create_uniform_map(num_bits=self.num_bits)
            self.norm_lookup_table = self.norm_lookup_table.to(device)
        else:
            raise NotImplementedError("Invalid quantization methods.")

    @staticmethod
    def create_uniform_map(symmetric=False, num_bits=4):
        if symmetric:
            # print("symmetric uniform quantization")
            negative = torch.linspace(-1, 0, 2 ** (num_bits - 1))
            positive = torch.linspace(0, 1, 2 ** (num_bits - 1))
            table = torch.cat([negative, positive[1:]])
        else:
            # print("asymmetric uniform quantization")
            table = torch.linspace(-1, 1, 2**num_bits)
        return table
    
    @staticmethod
    def create_normal_map(offset=0.9677083, symmetric=False, num_bits=2):
        try:
            from scipy.stats import norm
        except ImportError:
            raise ImportError("The required package 'scipy' is not installed. Please install it to continue.")

        variations = 2**num_bits
        if symmetric:
            v = norm.ppf(torch.linspace(1 - offset, offset, variations + 1)).tolist()
            values = []
            for index in range(len(v) - 1):
                values.append(0.5 * v[index] + 0.5 * v[index + 1])
            v = values
        else:
            # one more positive value, this is an asymmetric type
            v1 = norm.ppf(torch.linspace(offset, 0.5, variations // 2 + 1)[:-1]).tolist()
            v2 = [0]
            v3 = (-norm.ppf(torch.linspace(offset, 0.5, variations // 2)[:-1])).tolist()
            v = v1 + v2 + v3

        values = torch.Tensor(v)
        values = values.sort().values
        values /= values.max()
        return values
    
    def quantize_block(self, weight, num_std=2.5):
        if len(weight.shape) != 2:
            raise ValueError(f"Only support 2D matrix, but your input has {len(weight.shape)} dimensions.")
        if weight.shape[0] * weight.shape[1] % self.block_size != 0:
            raise ValueError(
                f"Weight with shape ({weight.shape[0]} x {weight.shape[1]}) "
                f"is not dividable by block size {self.block_size}."
            )

        M, N = weight.shape
        device = weight.device

        # Quantization
        weight_flatten = weight.flatten()  # (M*N, )
        weight_block = weight_flatten.reshape(-1, self.block_size)  # (L, B), L = M * N / B
        if self.method == "normal":
            weight_max = weight_block.abs().max(dim=-1)[0]  # (L, 1)
        elif self.method == "uniform":
            weight_max = weight_block.mean(dim=-1) + num_std * weight_block.std(dim=-1)
        else:
            raise NotImplementedError("Method not supported yet.")
        weight_max = weight_max.unsqueeze(-1)
        weight_divabs = weight_block / weight_max  # (L, B)
        weight_divabs = weight_divabs.unsqueeze(-1)  # (L, B, 1)
        L_reshaped = self.norm_lookup_table.reshape(1, -1)  # (1, 2**K)

        abs_diff = torch.abs(weight_divabs - L_reshaped)  # (L, B, 2**K)
        qweight = torch.argmin(abs_diff, dim=-1)  # (L, B)

        # Pack multiple k-bit into uint8
        qweight = qweight.reshape(-1, 8 // self.num_bits)
        qweight_pack = torch.zeros((M * N // 8 * self.num_bits, 1), dtype=torch.uint8, device=device)

        # data format example:
        # [1, 0, 3, 2] or [01, 00, 11, 10]  -> [10110001], LIFO
        for i in range(8 // self.num_bits):
            qweight[:, i] = qweight[:, i] << i * self.num_bits
            qweight_pack[:, 0] |= qweight[:, i]

        return qweight_pack, weight_max, weight.shape

    def dequantize_block(self, qweight, weight_max, weight_shape):
        # unpack weight
        device = qweight.device
        weight = torch.zeros((qweight.shape[0], 8 // self.num_bits), dtype=torch.float32, device=device)
        for i in range(8 // self.num_bits):
            lookup_table_idx = qweight.to(torch.long) % 2**self.num_bits  # get the most right 2 bits
            lookup_table_idx = lookup_table_idx.to(torch.long)
            weight[:, i] = self.norm_lookup_table[lookup_table_idx].squeeze()
            qweight = qweight >> self.num_bits  # right shift 2 bits of the original data

        weight_block = weight.reshape(-1, self.block_size)
        weight = weight_block * weight_max
        weight = weight.reshape(weight_shape)

        return weight
    
def _low_rank_decomposition(weight, reduced_rank=16):
    """
    :param weight: The matrix to decompose, of shape (H, W) :param reduced_rank: the final rank :return:
    """
    matrix_dimension = len(weight.size())
    if matrix_dimension != 2:
        raise ValueError(f"Only support 2D matrix, but your input has {matrix_dimension} dimensions.")

    # Use SVD to decompose a matrix, default full_matrices is False to save parameters
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

    L = U @ (torch.sqrt(torch.diag(S)[:, 0:reduced_rank]))
    R = torch.sqrt(torch.diag(S)[0:reduced_rank, :]) @ Vh

    return L, R
    # return {"L": L, "R": R, "U": U, "S": S, "Vh": Vh, "reduced_rank": reduced_rank}

def quantize_layer(weight: Union[torch.Tensor, nn.Parameter], num_bits: int, reduced_rank: int, num_iter=1, method="uniform"):
    dtype = weight.dtype
    
    weight = weight.to(device=compute_device, dtype=torch.float32)
    res = weight.clone()
    
    quantizer = BlockQuantizer(num_bits=num_bits, device=compute_device, method=method, block_size=64)
    
    for i in range(num_iter):
        clear_device_cache()
        
        quantized_weight, max_abs, shape = quantizer.quantize_block(res)
        dequantized_weight = quantizer.dequantize_block(quantized_weight, max_abs, shape)
    
        res = weight - dequantized_weight
    
        L, R = _low_rank_decomposition(res, reduced_rank=reduced_rank)
        res = weight - torch.mm(L, R)
    
    lora_A, lora_B = R, L
    
    return dequantized_weight.to(device=compute_device, dtype=dtype), lora_A, lora_B


def convert_linear_layer(
    model: nn.Module, 
    quantization_bits: int = 4,
    rank: int = 16,
    num_iters: int = 5,
    quantization_method: str = "uniform",
    target_modules: List[str] = ['all-linear'],
    excluded_modules: List[str] = ['classifier']
) -> nn.Module:
    """
    Convert a torch.nn.Linear layer in to a custom LoraLinearLayer including quantizing and computing the low-rank decomposition  
    """
    all_linear = len(target_modules) == 1 and target_modules[0] == 'all-linear'
    for name, module in model.named_children():
        
        if any(blocked in name for blocked in excluded_modules):
            continue
        
        if (all_linear and isinstance(module, nn.Linear)) or any(targetted in name for targetted in target_modules):
            print(f"Converting {name} layer")
            loftq_layer = LoraLinearLayer(
                    module,
                    quantization_bits=quantization_bits,
                    reduced_rank=rank,
                    num_iters=num_iters,
                    quantization_method=quantization_method
                )
            loftq_layer.quantize()
            
            setattr(
                model,
                name,
                loftq_layer    
            )
        else:
            convert_linear_layer(
                module,
                quantization_bits=quantization_bits,
                rank=rank,
                num_iters=num_iters,
                quantization_method=quantization_method,
                target_modules=target_modules,
                excluded_modules=excluded_modules
            )
    return model
    

def requantize_linear_layer(
    model: nn.Module, 
    target_modules: List[str] = ['all-linear'],
    excluded_modules: List[str] = ['classifier'],
    pre_quantized_weights: Dict = None
) -> nn.Module:
    """
    Convert a torch.nn.Linear layer in to a custom LoraLinearLayer and reapplies precomputed quantized values and low-rank decomposition
    """
    all_linear = len(target_modules) == 1 and target_modules[0] == 'all-linear'
    to_update = []
    for name, module in model.named_modules():
        print(name)
        if any(blocked in name for blocked in excluded_modules):
            continue
        
        if (all_linear and isinstance(module, nn.Linear)) or any(targetted in name for targetted in target_modules):
            to_update.append((name, module))
            print(f"Converting {name} layer")
    
    for name, module in to_update:
        setattr(
            model,
            name,
            LoraLinearLayer(
                module,
                pre_quantized_weights=pre_quantized_weights[name]
            )
        )

    return model
    