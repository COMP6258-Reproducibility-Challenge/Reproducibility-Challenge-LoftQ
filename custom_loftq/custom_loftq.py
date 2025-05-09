import sys
from typing import Any, Dict, List, Tuple, Union # Added Tuple and Union for type hints

import torch
from torch import nn
import torch.nn.functional as F # Added for F.pad if used in BlockQuantizer
from accelerate.utils.memory import clear_device_cache
# from scipy.stats import norm # Conditionally imported in BlockQuantizer

# Define compute_device (ensure this is consistent with your setup)
if torch.cuda.is_available():
    compute_device = torch.device("cuda")
elif torch.backends.mps.is_available(): # For Apple Silicon
    compute_device = torch.device("mps")
else:
    compute_device = torch.device("cpu")

# HELPER FUNCTIONS (Previously in utils.py, moved here to break circular imports)
def index_dim(tensor: torch.Tensor, dim: int, start: int, block_size: int) -> torch.Tensor:
    """
    Indexes the tensor along a specified dimension.
    """
    slices = [slice(None)] * tensor.dim()
    slices[dim] = slice(start, start + block_size)
    return tensor[tuple(slices)]

def max_dim(m: torch.Tensor) -> Tuple[int, int]:
    """
    Finds the dimension index (from end, negative) with the most elements and its size.
    """
    if m.dim() == 0: 
        return 0, 0 
    if m.dim() == 1:
        return -1, m.shape[0]
    
    dim_idx_from_start = torch.argmax(torch.tensor(m.shape)).item()
    max_elements = m.shape[dim_idx_from_start]
    dim_idx_from_end = dim_idx_from_start - m.dim()
    
    return dim_idx_from_end, max_elements

# --- BlockQuantizer ---
class BlockQuantizer:
    def __init__(self, num_bits: int = 2, method: str = "uniform", block_size: int = 64, device: Union[str, torch.device] = "cpu", *args: Any, **kwargs: Any):
        # super().__init__(*args, **kwargs) # Uncomment if BlockQuantizer has a parent class
        self.num_bits = num_bits
        self.method = method
        self.block_size = block_size
        
        self.current_device = torch.device(device) if isinstance(device, str) else device

        if self.method == "normal":
            self.norm_lookup_table = self.create_normal_map(num_bits=self.num_bits)
        elif self.method == "uniform":
            self.norm_lookup_table = self.create_uniform_map(num_bits=self.num_bits)
        else:
            raise NotImplementedError(f"Quantization method '{method}' not implemented.")
        self.norm_lookup_table = self.norm_lookup_table.to(self.current_device)

    @staticmethod
    def create_uniform_map(symmetric: bool = False, num_bits: int = 4) -> torch.Tensor:
        if symmetric:
            negative_half_points = 2 ** (num_bits - 1)
            positive_half_points = 2 ** (num_bits - 1)
            negative = torch.linspace(-1, 0, negative_half_points, dtype=torch.float32)
            positive = torch.linspace(0, 1, positive_half_points, dtype=torch.float32)
            table = torch.cat([negative, positive[1:]]) 
        else:
            table = torch.linspace(-1, 1, 2**num_bits, dtype=torch.float32)
        return table
    
    @staticmethod
    def create_normal_map(offset: float = 0.9677083, symmetric: bool = False, num_bits: int = 2) -> torch.Tensor:
        try:
            from scipy.stats import norm
        except ImportError:
            raise ImportError("The required package 'scipy' is not installed. Please install it to use 'normal' quantization.")

        variations = 2**num_bits
        if symmetric:
            v_ppf = torch.linspace(1 - offset, offset, variations + 1, dtype=torch.float64) # Use float64 for precision with norm.ppf
            v = norm.ppf(v_ppf.clamp(1e-7, 1-1e-7).numpy()) 
            values = []
            for index in range(len(v) - 1):
                values.append(0.5 * v[index] + 0.5 * v[index + 1])
            v_tensor = torch.Tensor(values).float()
        else: 
            quantiles = torch.linspace(1e-5, 1.0 - 1e-5, variations, dtype=torch.float64)
            v_tensor = torch.from_numpy(norm.ppf(quantiles.numpy())).float()

        if v_tensor.abs().max() > 1e-7: 
             v_tensor = v_tensor / v_tensor.abs().max() 
        return v_tensor.sort().values

    @staticmethod
    def safe_subtract_argmin(a: torch.Tensor, b: torch.Tensor, block_size_safe_sub: int = 4096) -> torch.Tensor:
        # This function's logic for concatenation and handling various input shapes for 'a' and 'b'
        # can be complex. The version from your `loftq.py` (uploaded) was used as a base.
        # Ensure it robustly handles broadcasting between a_chunk and b_chunk.
        # The b_chunk is typically the lookup table (1, num_centroids).
        # a_chunk is (num_elements_in_block, 1) or (num_elements_in_block, features_if_any).
        # The subtraction `a_chunk.unsqueeze(-1) - b_chunk.view(1, -1)` implies b_chunk is 1D.

        if a.numel() == 0 or b.numel() == 0: 
            if a.numel() > 0 :
                out_shape_list = list(a.shape)
                if out_shape_list: out_shape_list[-1] = 1 # Placeholder for argmin's effect
                return torch.empty(out_shape_list, dtype=torch.long, device="cpu")
            return torch.empty(0, dtype=torch.long, device="cpu")

        if a.numel() < b.numel(): 
            return BlockQuantizer.safe_subtract_argmin(-b, -a, block_size_safe_sub)

        max_dim_A_idx, max_num_A = max_dim(a) 
        max_dim_B_idx, _ = max_dim(b) # Not directly used if b is small (lookup table)

        result_list = []
        for start in range(0, max_num_A, block_size_safe_sub):
            actual_block_size_A = min(block_size_safe_sub, max_num_A - start)
            if a.numel() == 1: 
                a_chunk = a 
            else:
                a_chunk = index_dim(a, max_dim_A_idx, start, actual_block_size_A)
            
            # b_chunk is typically the lookup table, not chunked.
            b_chunk_for_op = b.view(1, -1) # Ensure b is [1, num_centroids] for broadcasting
            if a_chunk.dim() == 1: a_chunk_for_op = a_chunk.unsqueeze(-1) # Make a_chunk [N, 1]
            else: a_chunk_for_op = a_chunk # If a_chunk is already [N, M] (e.g. from reshape)

            temp = torch.argmin(torch.abs(a_chunk_for_op - b_chunk_for_op), dim=-1) 
            result_list.append(temp.to("cpu"))

        if not result_list:
            if a.numel() > 0 :
                # Shape of result after argmin on last dim of (a_chunk_for_op - b_chunk_for_op)
                # If a_chunk_for_op was (N,1), result is (N). If (N,M), result is (N).
                out_shape_list = list(a.shape[:-1]) if a.dim() > 1 else [a.shape[0]]
                return torch.empty(out_shape_list, dtype=torch.long, device="cpu")
            return torch.empty(0, dtype=torch.long, device="cpu")

        # Determine concatenation dimension carefully
        # If a was 1D, max_dim_A_idx is -1. cat_dim_positive becomes 0.
        # If result_list[0] is also 1D (which it should be after argmin), cat(..., dim=0) is fine.
        cat_dim_positive = 0
        if a.dim() > 1 : # If original 'a' was multi-dimensional
            cat_dim_positive = max_dim_A_idx if max_dim_A_idx >=0 else max_dim_A_idx + a.dim()
            # If the chunking dimension was the one that got reduced by argmin, adjust
            if cat_dim_positive == a.dim() -1 and a_chunk_for_op.shape[-1] == b_chunk_for_op.shape[-1] :
                 # This case implies last dim of 'a' matched last dim of 'b_chunk_for_op' for subtraction,
                 # and argmin reduced it. So, we cat along a pre-existing dimension of 'a'.
                 if result_list[0].dim() <= cat_dim_positive and result_list[0].dim() > 0:
                     cat_dim_positive = result_list[0].dim() -1


        if result_list[0].dim() == 0 and len(result_list) > 1: 
            concatenated_result = torch.stack(result_list, dim=0) 
        elif result_list[0].dim() == 0 and len(result_list) == 1:
            concatenated_result = result_list[0] 
        elif result_list[0].dim() > 0 and (all(r.shape == result_list[0].shape for r in result_list) or cat_dim_positive < result_list[0].dim()):
            concatenated_result = torch.cat(result_list, dim=cat_dim_positive)
        else: 
            try: # Fallback for potentially tricky shapes
                concatenated_result = torch.stack(result_list, dim=0) # Stack if cat fails
            except RuntimeError:
                print(f"Warning: Could not concatenate or stack results in safe_subtract_argmin. Shapes: {[r.shape for r in result_list]}, cat_dim: {cat_dim_positive}")
                return result_list[0] if len(result_list) == 1 else torch.cat([r.flatten() for r in result_list]) # Last resort flatten and cat


        return concatenated_result


    def quantize_block(self, weight_2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        if weight_2d.dim() != 2:
            raise ValueError(f"Input weight must be 2D. Got {weight_2d.dim()} dimensions.")
        if weight_2d.numel() == 0:
            return torch.empty(0, dtype=torch.uint8, device=self.current_device), \
                   torch.empty(0, dtype=weight_2d.dtype, device=self.current_device), \
                   weight_2d.shape

        original_shape = weight_2d.shape
        device = weight_2d.device 
        self.norm_lookup_table = self.norm_lookup_table.to(device)

        weight_flat = weight_2d.flatten()
        num_total_elements = weight_flat.numel()
        
        num_blocks_total = (num_total_elements + self.block_size - 1) // self.block_size
        padded_total_size = num_blocks_total * self.block_size
        padding_needed_total = padded_total_size - num_total_elements

        if padding_needed_total > 0:
            weight_flat_padded = F.pad(weight_flat, (0, padding_needed_total))
        else:
            weight_flat_padded = weight_flat
        
        weight_blocks_padded = weight_flat_padded.reshape(num_blocks_total, self.block_size)

        if self.method == "normal":
            scales = weight_blocks_padded.abs().max(dim=-1, keepdim=False)[0] 
        elif self.method == "uniform":
            scales = weight_blocks_padded.abs().max(dim=-1, keepdim=False)[0]
        else:
            raise NotImplementedError("Method not supported yet.")
        
        scales = torch.where(scales == 0, torch.tensor(1.0, device=device, dtype=scales.dtype), scales)
        
        scaled_weights = weight_blocks_padded / scales.unsqueeze(-1)
        
        qweight_indices = torch.argmin(torch.abs(scaled_weights.unsqueeze(-1) - self.norm_lookup_table.view(1, 1, -1)), dim=-1)
        qweight_indices_flat = qweight_indices.flatten()

        if self.num_bits < 8:
            bits_per_byte = 8 // self.num_bits
            qweight_reshaped = qweight_indices_flat.reshape(-1, bits_per_byte)
            qweight_pack = torch.zeros((qweight_reshaped.shape[0], 1), dtype=torch.uint8, device=device)
            for i in range(bits_per_byte):
                qweight_pack[:, 0] |= qweight_reshaped[:, i].to(torch.uint8) << (i * self.num_bits)
        elif self.num_bits == 8:
            qweight_pack = qweight_indices_flat.to(torch.uint8).unsqueeze(-1)
        else:
            qweight_pack = qweight_indices_flat.to(torch.int32) # Or other appropriate type for num_bits > 8

        return qweight_pack, scales, original_shape

    def dequantize_block(self, qweight_pack: torch.Tensor, scales: torch.Tensor, target_original_shape: Tuple[int, int]) -> torch.Tensor:
        if qweight_pack.numel() == 0:
            return torch.empty(target_original_shape, dtype=scales.dtype, device=qweight_pack.device)

        device = qweight_pack.device
        self.norm_lookup_table = self.norm_lookup_table.to(device)
        scales = scales.to(device)

        if self.num_bits < 8:
            bits_per_byte = 8 // self.num_bits
            mask = (1 << self.num_bits) - 1
            unpacked_indices_list = []
            for i in range(bits_per_byte):
                shift = i * self.num_bits
                indices = (qweight_pack.squeeze(-1) >> shift) & mask 
                unpacked_indices_list.append(indices)
            qweight_indices_flat = torch.stack(unpacked_indices_list, dim=-1).view(-1).long()
        elif self.num_bits == 8:
            qweight_indices_flat = qweight_pack.squeeze(-1).long()
        else:
            qweight_indices_flat = qweight_pack.long() # Assuming direct indices for num_bits > 8

        dequantized_flat_padded_normalized = self.norm_lookup_table[qweight_indices_flat]
        
        num_blocks_total = scales.shape[0]
        if num_blocks_total == 0: # Handle empty scales
            return torch.empty(target_original_shape, dtype=scales.dtype, device=device)
            
        block_size_used = dequantized_flat_padded_normalized.numel() // num_blocks_total
        
        dequantized_blocks_normalized = dequantized_flat_padded_normalized.reshape(num_blocks_total, block_size_used)
        scaled_blocks = dequantized_blocks_normalized * scales.unsqueeze(-1)
        
        scaled_flat_padded = scaled_blocks.view(-1)
        original_numel = target_original_shape[0] * target_original_shape[1]
        dequantized_flat = scaled_flat_padded[:original_numel]
        
        return dequantized_flat.reshape(target_original_shape)

# --- LoftQ Linear Layers ---
class BaseLoftqLinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        quantization_bits: int = 4,
        reduced_rank: int = 16,
        num_iters: int = 1, 
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
        
        # Ensure reduced_rank is not larger than dimensions
        actual_rank_for_lora_A = min(reduced_rank, self.in_features)
        actual_rank_for_lora_B = min(reduced_rank, self.out_features)
        # LoRA matrices usually share the same rank 'r'
        self.actual_lora_rank = min(reduced_rank, self.in_features, self.out_features)
        if self.actual_lora_rank == 0 and reduced_rank > 0: # If dimensions are 0, rank must be 0
            print(f"Warning: Reduced rank forced to 0 due to zero in_features/out_features for LoRA layers.")


        self.lora_A = nn.Linear(self.in_features, self.actual_lora_rank, bias=False)
        self.lora_B = nn.Linear(self.actual_lora_rank, self.out_features, bias=False)
        
        self.quantization_bits = quantization_bits
        self.reduced_rank = reduced_rank # Store the requested rank
        self.num_iters = num_iters 
        self.quantization_method = quantization_method
        self.quantizer = BlockQuantizer(
            num_bits=quantization_bits, 
            device=compute_device, 
            method=quantization_method, 
            block_size=64 
        )
        
    def quantize(self, weight: torch.Tensor):
        raise NotImplementedError
    
    def _loftq(self, weight_2d: torch.Tensor, reduced_rank_req: int, num_iter: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int,int], torch.Tensor, torch.Tensor]:
        current_dtype = weight_2d.dtype
        weight_2d_float32 = weight_2d.clone().to(device=compute_device, dtype=torch.float32)
        
        q_packed_base, scales_base, _ = self.quantizer.quantize_block(weight_2d_float32)
        dequantized_base_2d_float32 = self.quantizer.dequantize_block(q_packed_base, scales_base, weight_2d_float32.shape)
    
        residual_for_svd_float32 = weight_2d_float32 - dequantized_base_2d_float32
        
        actual_reduced_rank_svd = min(reduced_rank_req, residual_for_svd_float32.shape[0], residual_for_svd_float32.shape[1])
        if actual_reduced_rank_svd < reduced_rank_req:
            print(f"Warning: SVD rank reduced from {reduced_rank_req} to {actual_reduced_rank_svd} due to matrix dimensions for SVD.")
        if actual_reduced_rank_svd == 0 : # If rank becomes 0, handle SVD appropriately
             L_factor_float32 = torch.zeros((residual_for_svd_float32.shape[0], 0), device=compute_device, dtype=torch.float32)
             R_factor_float32 = torch.zeros((0, residual_for_svd_float32.shape[1]), device=compute_device, dtype=torch.float32)
        else:
            L_factor_float32, R_factor_float32 = self._low_rank_decomposition(residual_for_svd_float32, actual_reduced_rank_svd) 
        
        return (q_packed_base, 
                dequantized_base_2d_float32.to(current_dtype), 
                scales_base, 
                weight_2d.shape, 
                R_factor_float32.to(current_dtype), 
                L_factor_float32.to(current_dtype)) 
    
    def _low_rank_decomposition(self, weight_2d_residual: torch.Tensor, reduced_rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if weight_2d_residual.dim() != 2:
            raise ValueError(f"Only support 2D matrix, but input has {weight_2d_residual.dim()} dimensions.")
        if reduced_rank <= 0: # Handle non-positive rank
            return torch.zeros((weight_2d_residual.shape[0], 0), device=weight_2d_residual.device, dtype=weight_2d_residual.dtype), \
                   torch.zeros((0, weight_2d_residual.shape[1]), device=weight_2d_residual.device, dtype=weight_2d_residual.dtype)

        U, S, Vh = torch.linalg.svd(weight_2d_residual, full_matrices=False)

        L_factor = U[:, :reduced_rank] @ torch.diag(S[:reduced_rank].sqrt())
        R_factor = torch.diag(S[:reduced_rank].sqrt()) @ Vh[:reduced_rank, :]
        
        return L_factor, R_factor

class TrueQuantizedLinear(BaseLoftqLinear):
    def quantize(self, weight: torch.Tensor): 
        W_initial_2d = weight.clone() 
        
        # The rank passed to _loftq should be the actual LoRA rank for SVD
        rank_for_svd = self.actual_lora_rank 

        qweight_pack, _, scales, shape_2d, r_factor, l_factor = self._loftq(
            W_initial_2d, 
            rank_for_svd, # Use actual_lora_rank for SVD
            self.num_iters
        )
        
        self.register_buffer('qweight', qweight_pack)
        self.register_buffer('weight_scales', scales) 
        self.register_buffer('weight_shape_buf', torch.tensor(shape_2d, device=compute_device))
        
        if self.actual_lora_rank > 0:
            self.lora_A.weight.data = r_factor.to(self.lora_A.weight.dtype)
            self.lora_B.weight.data = l_factor.to(self.lora_B.weight.dtype)
        else: # Handle zero rank for LoRA layers if SVD rank was 0
            self.lora_A.weight.data.zero_()
            self.lora_B.weight.data.zero_()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_2d_shape = tuple(self.weight_shape_buf.tolist())
        
        dequantized_weight = self.quantizer.dequantize_block(
            self.qweight, 
            self.weight_scales, 
            original_2d_shape 
        )
        
        base_output = F.linear(x, dequantized_weight, self.bias)
        if self.actual_lora_rank > 0:
            lora_output = self.lora_B(self.lora_A(x))
            return base_output + lora_output
        return base_output
        
class LoraLinearLayer(BaseLoftqLinear):
    def __init__(
        self,
        original_base_layer: nn.Linear, 
        quantization_bits: int = 4,
        reduced_rank: int = 16,
        num_iters: int = 1,
        quantization_method: str = "uniform"
    ):
        dequantized_internal_base = nn.Linear(
            original_base_layer.in_features, 
            original_base_layer.out_features, 
            bias=(original_base_layer.bias is not None)
        )
        if original_base_layer.bias is not None:
            dequantized_internal_base.bias.data = original_base_layer.bias.data.clone()
        
        super().__init__(
            dequantized_internal_base, 
            quantization_bits,
            reduced_rank, # BaseLoftqLinear will calculate self.actual_lora_rank
            num_iters,
            quantization_method
        )
        
        self.internal_dequantized_base_layer = dequantized_internal_base

        for param in self.internal_dequantized_base_layer.parameters():
            param.requires_grad = False
            
    def quantize(self, original_linear_layer_weight: torch.Tensor):
        W_initial_2d = original_linear_layer_weight.clone()
        
        # Use self.actual_lora_rank which was determined in BaseLoftqLinear.__init__
        rank_for_svd = self.actual_lora_rank

        _, dequantized_base_2d, _, _, r_factor, l_factor = self._loftq(
            W_initial_2d, 
            rank_for_svd, 
            self.num_iters
        )
        
        self.internal_dequantized_base_layer.weight.data = dequantized_base_2d.to(
            self.internal_dequantized_base_layer.weight.dtype
        )

        if self.actual_lora_rank > 0:
            self.lora_A.weight.data = r_factor.to(self.lora_A.weight.dtype)
            self.lora_B.weight.data = l_factor.to(self.lora_B.weight.dtype)
        else:
            self.lora_A.weight.data.zero_()
            self.lora_B.weight.data.zero_()

        
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        base_out = self.internal_dequantized_base_layer(x)
        if self.actual_lora_rank > 0:
            lora_delta = self.lora_B(self.lora_A(x))
            return base_out + lora_delta
        return base_out

# --- Model Conversion Function ---
def convert_linear_layer(
    model: nn.Module, 
    quantization_bits: int = 4,
    rank: int = 16,
    num_iters: int = 1, 
    quantization_method: str = "uniform",
    target_modules: List[str] = ['all-linear'], 
    excluded_modules: List[str] = ['classifier', 'lm_head'], 
    true_quantization: bool = False 
) -> nn.Module:
    all_linear_targeted = (len(target_modules) == 1 and target_modules[0].lower() == 'all-linear')
    
    for name, module in model.named_children():
        is_excluded = any(excluded_name.lower() in name.lower() for excluded_name in excluded_modules)
        if is_excluded:
            continue
        
        is_target_module_type = isinstance(module, nn.Linear)
        
        should_convert = False
        if is_target_module_type:
            if all_linear_targeted:
                should_convert = True
            else:
                # Check if any part of the target_modules list is a substring of the current module's name
                if any(target_name.lower() in name.lower() for target_name in target_modules):
                    should_convert = True
        
        if should_convert:
            print(f"Converting module: {name} ({type(module).__name__}) to LoftQ {'TrueQuantizedLinear' if true_quantization else 'LoraLinearLayer'}")
            original_linear_layer = module 
            
            if true_quantization:
                loftq_layer = TrueQuantizedLinear(
                    base_layer=original_linear_layer, 
                    quantization_bits=quantization_bits,
                    reduced_rank=rank,
                    num_iters=num_iters,
                    quantization_method=quantization_method
                )
                loftq_layer.quantize(original_linear_layer.weight.data)
            else: 
                loftq_layer = LoraLinearLayer(
                    original_base_layer=original_linear_layer, 
                    quantization_bits=quantization_bits,
                    reduced_rank=rank,
                    num_iters=num_iters,
                    quantization_method=quantization_method
                )
                loftq_layer.quantize(original_linear_layer.weight.data)
            
            setattr(model, name, loftq_layer)

        elif len(list(module.children())) > 0: 
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

# --- User-provided requantize_linear_layer ---
def requantize_linear_layer(
    model: nn.Module, 
    target_modules: List[str] = ['all-linear'],
    excluded_modules: List[str] = ['classifier'],
    pre_quantized_weights: Dict = None, # Dict[module_full_name, Dict[weight_name, tensor]]
    true_quantization: bool = False
) -> nn.Module:
    all_linear = len(target_modules) == 1 and target_modules[0].lower() == 'all-linear'
    
    # We need to iterate using named_modules to get the full path for matching pre_quantized_weights
    modules_to_process = []
    for name, module_instance in model.named_modules():
        parent_module = model
        child_name_for_setattr = name
        if '.' in name:
            path_parts = name.split('.')
            child_name_for_setattr = path_parts[-1]
            parent_path = '.'.join(path_parts[:-1])
            parent_module = model.get_submodule(parent_path)
        
        if any(blocked.lower() in name.lower() for blocked in excluded_modules):
            continue
        
        is_target_type = isinstance(module_instance, nn.Linear)
        should_process_flag = False
        if is_target_type:
            if all_linear:
                should_process_flag = True
            elif any(targeted.lower() in name.lower() for targeted in target_modules):
                should_process_flag = True
        
        if should_process_flag:
            modules_to_process.append((name, module_instance, parent_module, child_name_for_setattr))
            # print(f"Identified for requantization: {name}") # Debug print

    for name, module_to_replace, parent_for_setattr, child_attr_name in modules_to_process:
        if pre_quantized_weights is None or name not in pre_quantized_weights:
            print(f"Warning: No precomputed weights found for {name} in pre_quantized_weights dict. Skipping requantization for this layer.")
            continue
        
        weights_info = pre_quantized_weights[name]
        print(f"Requantizing module: {name} with {'true' if true_quantization else 'simulated'} quantization using precomputed weights.")
        
        q_bits = weights_info.get('quantization_bits', 4)
        r_rank = weights_info.get('reduced_rank', 16)
        q_method = weights_info.get('quantization_method', 'uniform')
        # num_iters is part of the _loftq process, not directly needed for loading precomputed weights into the layer structure.

        new_layer = None
        if true_quantization:
            if 'qweight' in weights_info and 'weight_scales' in weights_info:
                new_layer = TrueQuantizedLinear(
                    base_layer=module_to_replace, # Original nn.Linear to determine in/out features etc.
                    quantization_bits=q_bits,
                    reduced_rank=r_rank,
                    quantization_method=q_method
                )
                new_layer.register_buffer("qweight", weights_info['qweight'].to(compute_device))
                new_layer.register_buffer("weight_scales", weights_info['weight_scales'].to(compute_device))
                
                # Handle weight_shape_buf (consistent with TrueQuantizedLinear's own quantize method)
                if 'weight_shape_buf' in weights_info: # If explicitly saved
                     new_layer.register_buffer("weight_shape_buf", weights_info['weight_shape_buf'].to(compute_device))
                elif 'weight_shape' in weights_info: # User's original key
                     new_layer.register_buffer("weight_shape_buf", torch.tensor(weights_info['weight_shape'], device=compute_device))
                else: # Fallback: infer from original module if not saved
                    new_layer.register_buffer("weight_shape_buf", torch.tensor(module_to_replace.weight.shape, device=compute_device))
            else:
                print(f"Warning: Incomplete quantization info for {name} (missing qweight or weight_scales). Cannot create TrueQuantizedLinear.")
                continue
        else: # LoraLinearLayer (simulated)
            new_layer = LoraLinearLayer(
                original_base_layer=module_to_replace, # Original nn.Linear
                quantization_bits=q_bits,
                reduced_rank=r_rank,
                quantization_method=q_method
            )
            # For LoraLinearLayer, we set the dequantized base weight directly
            if 'dequantized_base_weight' in weights_info: # Assuming this key for the dequantized base
                new_layer.internal_dequantized_base_layer.weight.data = weights_info['dequantized_base_weight'].to(
                    new_layer.internal_dequantized_base_layer.weight.dtype).to(compute_device)
            else:
                print(f"Warning: Missing 'dequantized_base_weight' for LoraLinearLayer {name}. Base weights will be from its init (likely random).")


        if new_layer:
            # Load LoRA A and B weights (using consistent keys)
            # Assuming pre_quantized_weights stores them as 'lora_A_weight' and 'lora_B_weight'
            if 'lora_A_weight' in weights_info and new_layer.actual_lora_rank > 0 :
                new_layer.lora_A.weight.data = weights_info['lora_A_weight'].to(new_layer.lora_A.weight.dtype).to(compute_device)
            elif 'lora_A' in weights_info and new_layer.actual_lora_rank > 0: # Fallback to user's original key
                new_layer.lora_A.weight.data = weights_info['lora_A'].to(new_layer.lora_A.weight.dtype).to(compute_device)

            if 'lora_B_weight' in weights_info and new_layer.actual_lora_rank > 0:
                new_layer.lora_B.weight.data = weights_info['lora_B_weight'].to(new_layer.lora_B.weight.dtype).to(compute_device)
            elif 'lora_B' in weights_info and new_layer.actual_lora_rank > 0: # Fallback to user's original key
                new_layer.lora_B.weight.data = weights_info['lora_B'].to(new_layer.lora_B.weight.dtype).to(compute_device)
            
            # Load bias
            if new_layer.has_bias:
                if 'bias' in weights_info:
                    new_layer.bias.data = weights_info['bias'].to(new_layer.bias.dtype).to(compute_device)
                elif module_to_replace.bias is not None: # Fallback to original bias if not in saved weights
                    new_layer.bias.data = module_to_replace.bias.data.clone().to(compute_device)
            
            setattr(parent_for_setattr, child_attr_name, new_layer)
            
    return model

# --- User-provided analyze_model_memory ---
def analyze_model_memory(model: nn.Module):
    # Using user's provided structure for analyze_model_memory
    total_params_standard_layers = 0 # Params from non-LoftQ layers (nn.Linear, nn.Conv2d etc.)
    
    # For TrueQuantizedLinear
    truequant_orig_params_sum = 0 # Sum of original parameters for layers converted to TrueQuantizedLinear
    truequant_actual_bytes_sum = 0 # Actual storage for qweight, scales, bias
    
    # For LoraLinearLayer (simulated)
    lorasim_dequant_base_params_sum = 0 # Parameters of the dequantized base in LoraLinearLayer
    
    # For all LoftQ layers (TrueQuant or Simulated)
    lora_adapter_params_sum = 0 # Parameters for lora_A and lora_B

    processed_loftq_children = set() # To avoid double counting if a LoftQ layer is a child of another counted module

    for name, module in model.named_modules():
        if module in processed_loftq_children:
            continue

        if isinstance(module, TrueQuantizedLinear):
            processed_loftq_children.add(module)
            if hasattr(module, 'weight_shape_buf'): # From TrueQuantizedLinear
                orig_out, orig_in = module.weight_shape_buf.tolist()
                truequant_orig_params_sum += orig_in * orig_out
            if module.has_bias:
                truequant_orig_params_sum += module.out_features
                if module.bias is not None: # Bias is stored unquantized
                    truequant_actual_bytes_sum += module.bias.numel() * module.bias.element_size()
            
            if hasattr(module, 'qweight') and hasattr(module, 'weight_scales'):
                truequant_actual_bytes_sum += module.qweight.numel() * module.qweight.element_size()
                truequant_actual_bytes_sum += module.weight_scales.numel() * module.weight_scales.element_size()
            
            if hasattr(module, 'lora_A') and module.lora_A is not None:
                lora_adapter_params_sum += sum(p.numel() for p in module.lora_A.parameters())
            if hasattr(module, 'lora_B') and module.lora_B is not None:
                lora_adapter_params_sum += sum(p.numel() for p in module.lora_B.parameters())

        elif isinstance(module, LoraLinearLayer):
            processed_loftq_children.add(module)
            if hasattr(module, 'internal_dequantized_base_layer'):
                base = module.internal_dequantized_base_layer
                lorasim_dequant_base_params_sum += sum(p.numel() for p in base.parameters())
                # Conceptually, these were the params that got "quantized" then dequantized
                # truequant_orig_params_sum += base.in_features * base.out_features 
                # if base.bias is not None:
                #     truequant_orig_params_sum += base.out_features

            if hasattr(module, 'lora_A') and module.lora_A is not None:
                lora_adapter_params_sum += sum(p.numel() for p in module.lora_A.parameters())
            if hasattr(module, 'lora_B') and module.lora_B is not None:
                lora_adapter_params_sum += sum(p.numel() for p in module.lora_B.parameters())
            
        elif not list(module.children()): # Leaf module that is not LoftQ
            # This sums parameters of all other leaf modules (Conv, Norm, Embed, non-LoftQ Linear)
            total_params_standard_layers += sum(p.numel() for p in module.parameters())


    print(f"\n--- Model Memory Analysis (User's Version Logic) ---")
    print(f"TrueQuantized Layers (Original Param Size Sum): {truequant_orig_params_sum:,}")
    print(f"  - Actual Storage (qweight, scales, bias): {truequant_actual_bytes_sum / (1024**2):.2f} MB")
    
    print(f"LoraLinearLayer (Simulated - Dequantized Base Param Sum): {lorasim_dequant_base_params_sum:,}")
    print(f"  - Estimated Storage (FP16): {lorasim_dequant_base_params_sum * 2 / (1024**2):.2f} MB") # Assuming FP16 for dequant base

    print(f"LoRA Adapter Parameters (A & B for ALL LoftQ layers): {lora_adapter_params_sum:,}")
    print(f"  - Estimated Storage (FP16): {lora_adapter_params_sum * 2 / (1024**2):.2f} MB")
    
    print(f"Other Standard Layers (Non-LoftQ, Leaf Modules): {total_params_standard_layers:,}")
    print(f"  - Estimated Storage (FP16): {total_params_standard_layers * 2 / (1024**2):.2f} MB")

    # Total Memory Calculation
    total_estimated_bytes = truequant_actual_bytes_sum + \
                              (lorasim_dequant_base_params_sum * 2) + \
                              (lora_adapter_params_sum * 2) + \
                              (total_params_standard_layers * 2)
    
    print(f"\nEstimated Total Model Memory Footprint: {total_estimated_bytes / (1024**2):.2f} MB")
    print("--- End Analysis ---")
    
    return {
        "truequant_orig_params_sum": truequant_orig_params_sum,
        "truequant_actual_bytes_sum": truequant_actual_bytes_sum,
        "lorasim_dequant_base_params_sum": lorasim_dequant_base_params_sum,
        "lora_adapter_params_sum": lora_adapter_params_sum,
        "total_params_standard_layers": total_params_standard_layers,
        "estimated_total_model_bytes": total_estimated_bytes
    }


if __name__ == '__main__':
    print(f"LoftQ module loaded. Running on: {compute_device}")

    print("\nTesting BlockQuantizer...")
    bq = BlockQuantizer(num_bits=4, method='uniform', device=compute_device)
    test_tensor = torch.randn(10, 128, device=compute_device) * 5
    q_w, scales, orig_s = bq.quantize_block(test_tensor)
    deq_w = bq.dequantize_block(q_w, scales, orig_s)
    print(f"BlockQuantizer Test: Original shape {orig_s}, Quantized pack shape {q_w.shape}, Scales shape {scales.shape}, Dequantized shape {deq_w.shape}")
    mse_loss = F.mse_loss(test_tensor, deq_w).item()
    if mse_loss < 0.1: 
        print(f"BlockQuantizer test: PASSED (approx). MSE: {mse_loss:.4f}")
    else:
        print(f"BlockQuantizer test: FAILED (approx). MSE: {mse_loss:.4f}")

    print("\nTesting TrueQuantizedLinear...")
    original_linear = nn.Linear(128, 256, bias=True).to(compute_device)
    true_q_linear = TrueQuantizedLinear(original_linear, reduced_rank=16, quantization_bits=4)
    true_q_linear.quantize(original_linear.weight.data)
    true_q_linear.to(compute_device) 
    
    dummy_input_linear = torch.randn(3, 128, device=compute_device)
    try:
        output_true_q = true_q_linear(dummy_input_linear)
        print(f"TrueQuantizedLinear forward pass successful. Output shape: {output_true_q.shape}")
    except Exception as e:
        print(f"Error in TrueQuantizedLinear forward: {e}")
        import traceback
        traceback.print_exc()

    print("\nTesting LoraLinearLayer...")
    original_linear_2 = nn.Linear(128, 256, bias=False).to(compute_device)
    lora_linear = LoraLinearLayer(original_linear_2, reduced_rank=8, quantization_bits=2, num_iters=1, quantization_method='normal')
    lora_linear.quantize(original_linear_2.weight.data)
    lora_linear.to(compute_device)

    try:
        output_lora_linear = lora_linear(dummy_input_linear)
        print(f"LoraLinearLayer forward pass successful. Output shape: {output_lora_linear.shape}")
    except Exception as e:
        print(f"Error in LoraLinearLayer forward: {e}")
        import traceback
        traceback.print_exc()

    print("\nTesting convert_linear_layer...")
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 20)
            self.activation = nn.ReLU()
            self.layer2_custom_name = nn.Linear(20, 5) 
            self.classifier = nn.Linear(5, 2) 

        def forward(self, x):
            x = self.activation(self.layer1(x))
            x = self.activation(self.layer2_custom_name(x))
            x = self.classifier(x)
            return x
    
    test_model_orig = TestModel().to(compute_device)
    import copy
    test_model_to_convert = copy.deepcopy(test_model_orig)

    converted_model = convert_linear_layer(
        test_model_to_convert,
        rank=4,
        true_quantization=True,
        target_modules=['layer1', 'layer2_custom_name'], 
        excluded_modules=['classifier']
    )
    print("Converted Model Structure:")
    print(converted_model)
    dummy_input_model = torch.randn(2, 10, device=compute_device)
    try:
        output_converted_model = converted_model(dummy_input_model)
        print(f"Converted model forward pass successful. Output shape: {output_converted_model.shape}")
    except Exception as e:
        print(f"Error in converted_model forward: {e}")
        import traceback
        traceback.print_exc()

    print("\nTesting analyze_model_memory (with user's version)...")
    analyze_model_memory(converted_model)
    
    print("\nTesting requantize_linear_layer (with user's version)...")
    dummy_pre_quant_weights = {}
    for name, module in converted_model.named_modules():
        if isinstance(module, TrueQuantizedLinear):
            dummy_pre_quant_weights[name] = {
                'qweight': module.qweight.clone(),
                'weight_scales': module.weight_scales.clone(),
                'weight_shape_buf': module.weight_shape_buf.clone(),
                'lora_A_weight': module.lora_A.weight.data.clone(),
                'lora_B_weight': module.lora_B.weight.data.clone(),
                'quantization_bits': module.quantization_bits,
                'reduced_rank': module.reduced_rank, # Storing the original requested rank
                'quantization_method': module.quantization_method,
            }
            if module.has_bias:
                dummy_pre_quant_weights[name]['bias'] = module.bias.data.clone()

    fresh_model_to_requant = TestModel().to(compute_device)
    requantized_model = requantize_linear_layer(
        fresh_model_to_requant,
        pre_quantized_weights=dummy_pre_quant_weights,
        true_quantization=True,
        target_modules=['layer1', 'layer2_custom_name'],
        excluded_modules=['classifier']
    )
    print("Requantized Model Structure:")
    print(requantized_model)
    try:
        output_requant_model = requantized_model(dummy_input_model)
        print(f"Requantized model forward pass successful. Output shape: {output_requant_model.shape}")
        if torch.allclose(output_converted_model, output_requant_model, atol=1e-5):
             print("Requantized model output matches converted model output: PASSED")
        else:
             print("Requantized model output MISMATCH.")
    except Exception as e:
        print(f"Error in requantized_model forward: {e}")
        import traceback
        traceback.print_exc()
