import sys
from typing import Any, Dict, List

import torch
from torch import nn
from accelerate.utils.memory import clear_device_cache

if torch.cuda.is_available():
    compute_device = torch.device("cuda")
elif torch.mps.is_available():
    compute_device = torch.device("mps")
else:
    compute_device = torch.device("cpu")

# import time

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
        """Apply LoftQ algorithm to get quantized weights and LoRA factors"""
        dtype = weight.dtype
        
        weight = weight.to(device=compute_device, dtype=torch.float32)
        res = weight.clone()
        
        for i in range(num_iter):
            clear_device_cache()
            
            # Quantize the residual
            quantized_weight, max_abs, shape = self.quantizer.quantize_block(res)
            dequantized_weight = self.quantizer.dequantize_block(quantized_weight, max_abs, shape)
        
            res = weight - dequantized_weight
        
            # Decompose the residual by SVD
            L, R = self._low_rank_decomposition(res, reduced_rank=reduced_rank)
            res = weight - torch.mm(L, R)
        
        # Return both the quantized representation and the LoRA factors
        return quantized_weight, dequantized_weight.to(device=compute_device, dtype=dtype), max_abs, shape, R, L
    
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

    @staticmethod
    def safe_subtract_argmin(a: torch.Tensor, b: torch.Tensor, block_size: int = 4096) -> torch.Tensor:
        """
        Perform memory-efficient version of the opearation argmin(abs(A - B)), chunking both A and B if needed, and
        storing the result on the CPU to minimize GPU memory usage. Note: if safe_broadcast_subtract() takes too long
        to compute, you can increase the block size to allow for more usage of GPU; if safe_broadcast_subtract() fails
        with CUDA out of memory, you can decrease the block size.

        Parameters
        ----------
        a : torch.Tensor (on CUDA)
            Left-hand tensor of subtraction. Must be at least 1D.
        b : torch.Tensor (on CUDA)
            Right-hand tensor. Must be broadcastable to A.
        block_size : int
            Max block size along the largest dimension. Default: 4096.
        """
        # Ensure A is always bigger than b
        if a.numel() < b.numel():
            return BlockQuantizer.safe_subtract_argmin(-b, -a, block_size)

        # dimension with max number of elements in A (in reverse order)
        # Number of elements in dimension max_dim_A
        max_dim_A, max_num_A = max_dim(a)

        # dimension with max number of elements in B (in reverse order)
        # Number of elements in dimension max_dim_B
        max_dim_B, max_num_B = max_dim(b)

        # List that stores the argmin blocks
        result_list = []

        for start in range(0, max_num_A, block_size):
            if a.numel() == 1:
                a_chunk = a.item()
            elif a.numel() == 0:
                break
            else:
                a_chunk = index_dim(a, max_dim_A, start, block_size)

            if b.numel() == 1:
                b_chunk = b.item()
            elif b.numel() == 0:
                break
            elif max_dim_B != max_dim_A:
                b_chunk = b
            elif max_dim_B == max_dim_A and (len(b.shape) < len(a.shape) or b.shape[max_dim_A] == 1):
                b_chunk = b
            else:
                b_chunk = index_dim(b, max_dim_B, start, block_size)

            # Subtract, compute abs(), compute argmin(), and move to CPU
            temp = torch.argmin(torch.abs(a_chunk - b_chunk), dim=-1, keepdim=True).to("cpu")
            result_list.append(temp)

        if len(result_list) == 0:
            return torch.zeros(0)

        return torch.squeeze(torch.cat(result_list, dim=max_dim_A))



    def quantize_block(self, weight, num_std=2.5):
        # start_time = time.time()
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
        weight = weight.flatten()  # (M*N, )
        weight_block = weight.reshape(-1, self.block_size)  # (L, B), L = M * N / B
        if self.method == "normal":
            weight_max = weight_block.abs().max(dim=-1)[0]  # (L, 1)
        elif self.method == "uniform":
            weight_max = weight_block.mean(dim=-1) + num_std * weight_block.std(dim=-1)
        else:
            raise NotImplementedError("Method not supported yet.")
        weight_max = weight_max.unsqueeze(-1)
        weight_divabs = weight_block / weight_max  # (L, B)
        del weight_block
        weight_max = weight_max.to("cpu")

        weight_divabs = weight_divabs.unsqueeze(-1)  # (L, B, 1)
        L_reshaped = self.norm_lookup_table.reshape(1, -1)  # (1, 2**K)
        # print(f"Tensor shapes: (L = {L_reshaped.shape}, weight = {weight_divabs.shape})")
        # print("performing safe broadcasting ... ")
        qweight = BlockQuantizer.safe_subtract_argmin(weight_divabs, L_reshaped, 128).to(self.device)
        # print(qweight.shape)
        # print("Done\n")
        del weight_divabs
        del L_reshaped

        # Convert bits
        if self.num_bits < 8:
            # Pack multiple k-bit into uint8
            bits_per_byte = 8 // self.num_bits
            qweight = qweight.reshape(-1, bits_per_byte)
            qweight_pack = torch.zeros((M * N // bits_per_byte, 1), dtype=torch.uint8, device=device)
            
            for i in range(bits_per_byte):
                qweight[:, i] = qweight[:, i] << i * self.num_bits
                qweight_pack[:, 0] |= qweight[:, i]
        elif self.num_bits == 8:
            # For 8-bit, direct conversion
            qweight_pack = qweight.byte().reshape(-1, 1)
        else:
            # For larger than 8 bits, use appropriate dtype
            if self.num_bits == 16:
                qweight_pack = qweight.short()
            elif self.num_bits == 32:
                qweight_pack = qweight.int()
            else:
                qweight_pack = qweight

        # end_time = time.time()
        # print(f"Time used = {(end_time - start_time) / 60} minutes")
        return qweight_pack, weight_max.to(self.device), (M, N)  # weight.shape



    def dequantize_block(self, qweight, weight_max, weight_shape):
        # unpack weight
        device = qweight.device
        if self.num_bits < 8:
            bits_per_byte = 8 // self.num_bits
            weight = torch.zeros((qweight.shape[0], bits_per_byte), dtype=torch.float32, device=device)
            for i in range(bits_per_byte):
                lookup_table_idx = qweight.to(torch.long) % 2**self.num_bits
                weight[:, i] = self.norm_lookup_table[lookup_table_idx].squeeze()
                qweight = qweight >> self.num_bits
        else:
            lookup_table_idx = qweight.squeeze().long()
            weight = self.norm_lookup_table[lookup_table_idx].reshape(-1, 1)

        weight_block = weight.reshape(-1, self.block_size)
        weight = weight_block * weight_max
        weight = weight.reshape(weight_shape)

        return weight
    
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