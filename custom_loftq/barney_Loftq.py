import sys
from typing import Any, Dict, List

import torch
import numpy as np
from scipy.stats import norm
from torch import nn
from accelerate.utils.memory import clear_device_cache

if torch.cuda.is_available():
    compute_device = torch.device("cuda")
elif torch.mps.is_available():
    compute_device = torch.device("mps")
else:
    compute_device = torch.device("cpu")
class BaseLoftqLinear(nn.Module):
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
        
        self.has_bias = base_layer.bias is not None
        if self.has_bias:
            self.bias = nn.Parameter(base_layer.bias.data.clone())
        else:
            self.register_parameter('bias', None)
        
        self.lora_A = nn.Linear(self.in_features, reduced_rank, bias=False)
        self.lora_B = nn.Linear(reduced_rank, self.out_features, bias=False)
        
        self.quantization_bits = quantization_bits
        self.reduced_rank = reduced_rank
        self.num_iters = num_iters
        self.quantization_method = quantization_method
        self.quantizer = BlockQuantizer(
            num_bits=quantization_bits, 
            device=compute_device, 
            method=quantization_method, 
            block_size=64
        )
        
    def quantize(self):
        pass

    def _loftq(self, weight, reduced_rank, num_iter):
        dtype = weight.dtype
        current_device = compute_device if 'compute_device' in globals() else self.device
        W_orig = weight.to(device=current_device, dtype=torch.float32)
        residual = W_orig.clone()

        adanf_n = 10
        adanf_c_start = 0.9
        adanf_c_end = 0.99
        adanf_p = 3

        if not hasattr(self, 'quantizer'):
             raise AttributeError("LoftQ layer requires a 'quantizer' attribute.")


        final_packed_qweights = None
        final_scales = None
        Q_t_last_iter = None
        A_final, B_final = None, None
        all_best_c_offsets_last_iter = None 


        for i in range(num_iter):
            clear_device_cache()
            Q_t = torch.zeros_like(residual) 

 
            current_iter_packed_qweights = []
            current_iter_scales = []
            current_iter_c_offsets = []


            if self.quantization_method == "adanf":
                num_elements = residual.numel()
                if num_elements % self.quantizer.block_size != 0:
                    raise ValueError(f"Residual size {num_elements} not divisible by block size {self.quantizer.block_size}")

                residual_blocks = residual.flatten().reshape(-1, self.quantizer.block_size)
                num_blocks = residual_blocks.shape[0]

                for block_idx in range(num_blocks):
                    W_group = residual_blocks[block_idx]
                    packed_qweight, scale, best_c_offset = run_adanf_grid_search(
                        W_group, self.quantizer, adanf_n, adanf_c_start, adanf_c_end, adanf_p
                    )
                    current_iter_packed_qweights.append(packed_qweight)
                    current_iter_scales.append(scale.unsqueeze(0))
                    current_iter_c_offsets.append(best_c_offset) 


                final_packed_qweights = torch.cat(current_iter_packed_qweights)
                final_scales = torch.cat(current_iter_scales)
                all_best_c_offsets_last_iter = current_iter_c_offsets 

                Q_t_blocks_dequant = []
                block_shape = torch.Size([self.quantizer.block_size])
                num_packed_per_block = self.quantizer.block_size // (8 // self.quantizer.num_bits)

                for block_idx in range(num_blocks):
                     best_map_for_block = self.quantizer.create_dynamic_normal_map(all_best_c_offsets_last_iter[block_idx])
                     start_idx = block_idx * num_packed_per_block
                     end_idx = (block_idx + 1) * num_packed_per_block
                     block_packed_qweight = final_packed_qweights[start_idx:end_idx]
                     block_scale = final_scales[block_idx]

                     dequantized_block = self.quantizer.dequantize_block(
                         block_packed_qweight, block_scale.unsqueeze(0), block_shape, lookup_map=best_map_for_block
                     )
                     Q_t_blocks_dequant.append(dequantized_block)

                Q_t = torch.cat(Q_t_blocks_dequant).reshape(W_orig.shape)


            elif self.quantization_method in ["normal", "uniform"]:
                packed_qweight, scales, shape = self.quantizer.quantize_block(residual, lookup_map=None)
                Q_t = self.quantizer.dequantize_block(packed_qweight, scales, shape, lookup_map=None)
                # LEAVE A COMMENT: Store results from this iteration for uniform/normal
                final_packed_qweights = packed_qweight
                final_scales = scales
            else:
                raise NotImplementedError(f"Quantization method {self.quantization_method} not supported in _loftq")

            Q_t_last_iter = Q_t 

            residual_after_quant = W_orig - Q_t
            L, R = self._low_rank_decomposition(residual_after_quant, reduced_rank=reduced_rank)
            residual = W_orig - torch.mm(L, R) 
            A_final, B_final = R, L 


        if final_packed_qweights is None or Q_t_last_iter is None or final_scales is None or A_final is None or B_final is None:
             raise RuntimeError("_loftq loop did not execute or failed to produce results.")

        return final_packed_qweights, Q_t_last_iter.to(dtype=dtype), final_scales, W_orig.shape, A_final.to(dtype=dtype), B_final.to(dtype=dtype)
    
    def _low_rank_decomposition(self, weight, reduced_rank=16):
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

#############################################################
#Grid Search Function
#############################################################
def run_adanf_grid_search(W_group, quantizer, n, c_start, c_end, p):
    best_c_offset = c_start
    min_error = float('inf')
    best_packed_qweight = None
    best_scale = None
    group_shape = W_group.shape
    device=W_group.device

    scale = W_group.abs().max()
    scale = torch.where(scale == 0, torch.tensor(1.0, device=W_group.device, dtype=scale.dtype), scale)

    if n <= 1:
         step_size = 0
         if n == 1:
             c_end = c_start
    else:
         step_size = (c_end - c_start) / (n - 1)

    for i in range(n):
        c_offset = c_start + step_size * i
        c_offset = max(0.50001, min(0.99999, c_offset))

        dynamic_map = quantizer.create_dynamic_normal_map(c_offset)

        W_group_norm = W_group / scale
        num_levels = dynamic_map.numel()
        L_reshaped = dynamic_map.reshape(1, num_levels) # Reshape for comparison
        abs_diff = torch.abs(W_group_norm.unsqueeze(-1) - L_reshaped) # W_group_norm needs unsqueeze
        qweight_indices = torch.argmin(abs_diff, dim=-1)

        dequantized_normalized = dynamic_map[qweight_indices]
        W_deq_group = dequantized_normalized * scale

        error = torch.linalg.norm(W_group.float() - W_deq_group.float(), ord=p).item()

        if error < min_error:
            min_error = error
            best_c_offset = c_offset

            num_indices_per_byte = 8 // quantizer.num_bits
            qweight_indices_1d = qweight_indices.flatten()
            qweight_reshaped = qweight_indices_1d.reshape(-1, num_indices_per_byte)
            packed_qweight = torch.zeros((qweight_reshaped.shape[0], 1), dtype=torch.uint8, device=device)
            for idx_pack in range(num_indices_per_byte):
                 packed_qweight[:, 0] |= qweight_reshaped[:, idx_pack].to(torch.uint8) << (idx_pack * quantizer.num_bits)

            best_packed_qweight = packed_qweight
            best_scale = scale

    if best_packed_qweight is None:
        raise RuntimeError(f"AdaNF grid search failed to find a best quantization for a block. n={n}")


    return best_packed_qweight, best_scale, best_c_offset
#############################################################


class TrueQuantizedLinear(BaseLoftqLinear):
        def quantize(self, weight: torch.Tensor):
            """Quantize weights and initialize LoRA adapters"""
            W_initial = weight.clone()
            
            # Apply LoftQ to get quantized weights and LoRA factors
            qweight_pack, dequantized_weight, weight_max, weight_shape, lora_A, lora_B = self._loftq(
                W_initial, 
                self.reduced_rank, 
                self.num_iters
            )
            
            # Store quantized weights as buffers (not parameters)
            self.register_buffer('qweight', qweight_pack)
            self.register_buffer('weight_max', weight_max)
            self.register_buffer('weight_shape', torch.tensor(weight_shape))
            
            # Initialize LoRA adapters
            self.lora_A.weight.data = lora_A
            self.lora_B.weight.data = lora_B

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass with on-the-fly dequantization"""
            # Dequantize weights
            weight = self.quantizer.dequantize_block(
                self.qweight, 
                self.weight_max, 
                (self.weight_shape[0].item(), self.weight_shape[1].item())
            )
            
            # Compute the base output using dequantized weights
            base_output = nn.functional.linear(x, weight, self.bias)
            
            # Apply LoRA adapters
            lora_output = self.lora_B(self.lora_A(x))
            
            return base_output + lora_output
        

class LoraLinearLayer(BaseLoftqLinear):
    def __init__(
        self,
        base_layer: nn.Linear,
        quantization_bits: int = 4,
        reduced_rank: int = 16,
        num_iters: int = 5,
        quantization_method: str = "uniform"
    ):
        super().__init__(
            base_layer,
            quantization_bits,
            reduced_rank,
            num_iters,
            quantization_method
        )
        
        self.base_layer = base_layer
        
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
    def quantize(self):
        W_initial = self.base_layer.weight.data.clone()
        
        qweight_pack, dequantized_weight, weight_max, weight_shape, lora_A, lora_B = self._loftq(W_initial, self.reduced_rank, self.num_iters)
        
        self.base_layer.weight.data = dequantized_weight
        self.lora_A.weight.data = lora_A
        self.lora_B.weight.data = lora_B
        
        del self.quantizer
        
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        out = self.base_layer(x, *args, **kwargs)
        lora = self.lora_B(self.lora_A(x))
        return out + lora


class BlockQuantizer:
    def __init__(self, num_bits=2, method="uniform", block_size=64, device="cpu", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_bits = num_bits
        self.device = device
        self.block_size = block_size
        self.method = method
        if self.method == "normal":
             # LEAVE A COMMENT: Original code created asymmetric map by default, preserving that here
            self.norm_lookup_table = self.create_normal_map(num_bits=self.num_bits, symmetric=False)
            self.norm_lookup_table = self.norm_lookup_table.to(device)
        elif self.method == "uniform":
             # LEAVE A COMMENT: Original code created asymmetric map by default, preserving that here
            self.norm_lookup_table = self.create_uniform_map(num_bits=self.num_bits, symmetric=False)
            self.norm_lookup_table = self.norm_lookup_table.to(device)
        elif self.method == "adanf":
            # LEAVE A COMMENT: Added retrieval of adanf_c_ref from kwargs
            self.c_ref = kwargs.get("adanf_c_ref", 0.995)
            self.q_c_ref = torch.tensor(norm.ppf(self.c_ref), device=self.device, dtype=torch.float32)
            if self.q_c_ref <= 0:
                 raise ValueError(f"Q(c_ref) must be positive, but got {self.q_c_ref} for c_ref={self.c_ref}. Ensure c_ref > 0.5.")
            pass
        else:
            raise NotImplementedError(f"Invalid quantization method: {self.method}")

    @staticmethod
    def create_uniform_map(symmetric=False, num_bits=4):
        variations = 2**num_bits
        if symmetric:
            if variations % 2 == 1:
                 midpoint = torch.tensor([0.0])
                 num_each_side = variations // 2
                 negative = torch.linspace(-1, 0, num_each_side + 1)[:-1]
                 positive = torch.linspace(0, 1, num_each_side + 1)[1:]
                 table = torch.cat([negative, midpoint, positive])
            else:
                 # LEAVE A COMMENT: Original code used linspace endpoints differently, adjusting for symmetry
                 negative = torch.linspace(-1, 0, variations // 2 + 1)[:-1]
                 positive = torch.linspace(0, 1, variations // 2 + 1)[1:]
                 table = torch.cat([negative, positive])
        else:
            table = torch.linspace(-1, 1, variations)
        return table.float()

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


    def create_dynamic_normal_map(self, c_offset):
        if not (0.5 < c_offset < 1.0):
            raise ValueError(f"c_offset must be between 0.5 and 1.0, but got {c_offset}")
        if not hasattr(self, 'q_c_ref'):
             raise AttributeError("BlockQuantizer not properly initialized for AdaNF. Missing 'q_c_ref'.")

        num_levels = 2**self.num_bits
        probabilities = torch.linspace(1 - c_offset, c_offset, num_levels)
        unnormalized_quantiles = torch.from_numpy(norm.ppf(probabilities.numpy())).to(dtype=torch.float32, device=self.device)
        dynamic_map = unnormalized_quantiles / self.q_c_ref
        return dynamic_map

    def quantize_block(self, weight, num_std=2.5, lookup_map = None):
        if len(weight.shape) != 2:
            raise ValueError(f"Only support 2D matrix, but your input has {len(weight.shape)} dimensions.")
        num_elements = weight.numel()
        if num_elements % self.block_size != 0:
            raise ValueError(
                f"Weight with shape {weight.shape} (total {num_elements} elements) "
                f"is not perfectly divisible by block size {self.block_size}."
            )

        M, N = weight.shape
        device = weight.device

        if self.method == "adanf":
            if lookup_map is None:
                raise ValueError("lookup_map must be provided when method is 'adanf'")
            current_lookup_table = lookup_map.to(device)
        elif self.method in ["normal", "uniform"]:
            if not hasattr(self, 'norm_lookup_table'):
                 raise AttributeError(f"Quantizer not initialized with a lookup table for method '{self.method}'")
            current_lookup_table = self.norm_lookup_table
        else:
             raise NotImplementedError(f"Quantization logic not implemented for method: {self.method}")

        weight_flatten = weight.flatten()
        weight_block = weight_flatten.reshape(-1, self.block_size)

        # LEAVE A COMMENT: Calculating weight_max using absmax for all relevant methods before normalization
        if self.method in ["normal", "adanf", "uniform"]:
            weight_max = weight_block.abs().max(dim=-1, keepdim=True)[0]
        else:
             raise NotImplementedError(f"Scaling factor logic missing for method: {self.method}")

        # LEAVE A COMMENT: Added check for division by zero
        weight_max = torch.where(weight_max == 0, torch.tensor(1.0, device=device, dtype=weight_max.dtype), weight_max)

        weight_divabs = weight_block / weight_max

        num_levels = current_lookup_table.numel()
        # LEAVE A COMMENT: Using current_lookup_table (which is the passed lookup_map for adanf) here
        L_reshaped = current_lookup_table.reshape(1, 1, num_levels)
        abs_diff = torch.abs(weight_divabs.unsqueeze(-1) - L_reshaped)
        qweight = torch.argmin(abs_diff, dim=-1)

        num_indices_per_byte = 8 // self.num_bits
        if 8 % self.num_bits != 0:
            raise ValueError(f"Packing requires 8 to be divisible by num_bits ({self.num_bits})")

        qweight_reshaped = qweight.reshape(-1, num_indices_per_byte)
        packed_qweight = torch.zeros((qweight_reshaped.shape[0], 1), dtype=torch.uint8, device=device)

        for i in range(num_indices_per_byte):
            packed_qweight[:, 0] |= qweight_reshaped[:, i].to(torch.uint8) << (i * self.num_bits)

        return packed_qweight, weight_max.squeeze(-1), weight.shape

    def dequantize_block(self, packed_qweight, weight_max, weight_shape, lookup_map=None):
        device = packed_qweight.device
        num_elements = weight_shape.numel()
        num_blocks = weight_max.shape[0]

        if num_elements % self.block_size != 0:
             raise ValueError(
                 f"Original weight shape {weight_shape} (total {num_elements} elements) "
                 f"is not perfectly divisible by block size {self.block_size}."
             )
        if num_blocks != num_elements // self.block_size:
             raise ValueError(f"Number of scaling factors ({num_blocks}) does not match number of blocks ({num_elements // self.block_size})")

        if self.method == "adanf":
            if lookup_map is None:
                raise ValueError("lookup_map must be provided when method is 'adanf'")
            current_lookup_table = lookup_map.to(device)
        elif self.method in ["normal", "uniform"]:
            if not hasattr(self, 'norm_lookup_table'):
                 raise AttributeError(f"Quantizer not initialized with a lookup table for method '{self.method}'")
            current_lookup_table = self.norm_lookup_table
        else:
             raise NotImplementedError(f"Dequantization logic not implemented for method: {self.method}")

        num_indices_per_byte = 8 // self.num_bits
        if 8 % self.num_bits != 0:
            raise ValueError(f"Unpacking requires 8 to be divisible by num_bits ({self.num_bits})")

        total_indices = num_elements
        unpacked_indices = torch.zeros(total_indices, dtype=torch.long, device=device)
        mask = (1 << self.num_bits) - 1

        for i in range(num_indices_per_byte):
            shift = i * self.num_bits
            indices_for_pos = (packed_qweight[:, 0] >> shift) & mask
            start_idx = i
            end_idx = total_indices
            step = num_indices_per_byte
            if indices_for_pos.shape[0] == unpacked_indices[start_idx:end_idx:step].shape[0]:
                 unpacked_indices[start_idx:end_idx:step] = indices_for_pos.to(torch.long)
            else:
                 raise RuntimeError(f"Shape mismatch during unpacking. Packed shape: {packed_qweight.shape}, Unpacked indices shape: {unpacked_indices.shape}")

        # LEAVE A COMMENT: Using current_lookup_table (which is the passed lookup_map for adanf) here
        dequantized_normalized_values = current_lookup_table[unpacked_indices]
        dequantized_normalized_block = dequantized_normalized_values.reshape(num_blocks, self.block_size)
        weight_max_reshaped = weight_max.unsqueeze(-1)
        dequantized_block = dequantized_normalized_block * weight_max_reshaped
        dequantized_weight = dequantized_block.reshape(weight_shape)

        return dequantized_weight.to(dtype=torch.float32)

def convert_linear_layer(
    model: nn.Module, 
    quantization_bits: int = 4,
    rank: int = 16,
    num_iters: int = 5,
    quantization_method: str = "uniform",
    target_modules: List[str] = ['all-linear'],
    excluded_modules: List[str] = ['classifier'],
    true_quantization: bool = False  # New parameter to enable true quantization
) -> nn.Module:
    """
    Convert a torch.nn.Linear layer in to a custom LoraLinearLayer including quantizing and computing the low-rank decomposition  
    """
    all_linear = len(target_modules) == 1 and target_modules[0] == 'all-linear'
    for name, module in model.named_children():
        
        if any(blocked in name for blocked in excluded_modules):
            continue
        
        if (all_linear and isinstance(module, nn.Linear)) or any(targetted in name for targetted in target_modules):
            print(f"Converting {name} layer with {'true' if true_quantization else 'simulated'} quantization")
            if true_quantization:
                loftq_layer = TrueQuantizedLinear(
                    module,
                    quantization_bits=quantization_bits,
                    reduced_rank=rank,
                    num_iters=num_iters,
                    quantization_method=quantization_method
                )
                
                loftq_layer.quantize(module.weight.data.clone())
            else:
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
                excluded_modules=excluded_modules,
                true_quantization=true_quantization
            )
    return model


def requantize_linear_layer(
    model: nn.Module, 
    target_modules: List[str] = ['all-linear'],
    excluded_modules: List[str] = ['classifier'],
    pre_quantized_weights: Dict = None,
    true_quantization: bool = False
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
            print(f"Will convert {name} layer with {'true' if true_quantization else 'simulated'} quantization")
    
    for name, module in to_update:
        if name not in pre_quantized_weights:
            print(f"Warning: No precomputed weights found for {name}, skipping")
            continue
        
        weights_info = pre_quantized_weights[name]
        
        if true_quantization:
            if 'qweight' in weights_info and 'weight_max' in weights_info:
                new_layer = TrueQuantizedLinear(
                    module,
                    quantization_bits=weights_info.get('quantization_bits', 4),
                    reduced_rank=weights_info.get('reduced_rank', 16),
                    quantization_method=weights_info.get('quantization_method', 'uniform')
                )
                
                print(weights_info['qweight'])
                sys.exit()
                new_layer.register_buffer("qweight", weights_info['qweight'])
                new_layer.register_buffer("weight_max", weights_info['weight_max'])
                new_layer.register_buffer("weight_shape", weights_info['weight_shape'])
            else:
                print(f"Warning: Incomplete quantization info for {name}, cannot create TrueQuantLinearLayer")
                continue
        else:
            new_layer = LoraLinearLayer(
                module,
                quantization_bits=weights_info.get('quantization_bits', 4),
                reduced_rank=weights_info.get('reduced_rank', 16),
                quantization_method=weights_info.get('quantization_method', 'uniform')
            )
            
            # Set the dequantized parameters
            if 'dequantized_weight' in weights_info:
                new_layer.base_layer.weight.data = weights_info['dequantized_weight']
            
        if 'lora_A' in weights_info and 'lora_B' in weights_info:
            new_layer.lora_A.weight.data = weights_info['lora_A']
            new_layer.lora_B.weight.data = weights_info['lora_B']
            
        if new_layer.has_bias and 'bias' in weights_info:
            new_layer.bias.data = weights_info['bias']
                
        # Replace the module
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        
        if parent_name:
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, new_layer)
        else:
            setattr(model, child_name, new_layer)

    return model

def analyze_model_memory(model):
    total_params = 0
    quant_params = 0
    quant_bytes = 0
    lora_params = 0
    other_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, TrueQuantizedLinear):
            # Quantized layer
            if hasattr(module, 'qweight'):
                quant_params += module.in_features * module.out_features
                quant_bytes += module.qweight.numel() * module.qweight.element_size()
            # LoRA adapters
            lora_params += module.lora_A.weight.numel() + module.lora_B.weight.numel()
        elif isinstance(module, LoraLinearLayer):
            if hasattr(module, 'base_layer'):
                param_count = module.base_layer.weight.numel()
                total_params += param_count
                other_params += param_count
            # LoRA adapters
            lora_params += module.lora_A.weight.numel() + module.lora_B.weight.numel()
        elif isinstance(module, nn.Linear):
            # Regular linear layer
            if hasattr(module, 'weight'):
                param_count = module.weight.numel()
                total_params += param_count
                other_params += param_count
    
    print(f"Quantized parameters: {quant_params:,} ({quant_bytes/(1024**2):.2f}MB)")
    print(f"LoRA parameters: {lora_params:,} ({lora_params*2/(1024**2):.2f}MB)")
    print(f"Other parameters: {other_params:,} ({other_params*2/(1024**2):.2f}MB)")
    print(f"Total: {other_params + quant_params + lora_params:,} ({((other_params*2) + quant_bytes + (lora_params*2))/(1024**2):.2f}MB)")
    