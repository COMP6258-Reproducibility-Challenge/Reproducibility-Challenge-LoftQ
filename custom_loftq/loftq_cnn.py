import torch
import torch.nn as nn
import torch.nn.functional as F
# from scipy.stats import norm # Required for 'normal' method in BlockQuantizer

# Define compute_device (ensure this is consistent with your setup)
if torch.cuda.is_available():
    compute_device = torch.device("cuda")
elif torch.backends.mps.is_available(): # For Apple Silicon
    compute_device = torch.device("mps")
else:
    compute_device = torch.device("cpu")

# --- BlockQuantizer (assuming this is part of your existing loftq.py or utils) ---
# For completeness, I'm including a version of it here.
# Ensure this matches or is adapted from your existing BlockQuantizer.
class _BlockQuantizer: # Renamed to avoid potential clash if user has BlockQuantizer
    def __init__(self, num_bits=4, method="uniform", block_size=64, device="cpu", *args, **kwargs):
        self.num_bits = num_bits
        self.device = device
        self.method = method
        self.block_size = block_size
        if self.method == "uniform":
            self.norm_lookup_table = self._create_uniform_map(num_bits=self.num_bits).to(device)
        elif self.method == "normal":
            self.norm_lookup_table = self._create_normal_map(num_bits=self.num_bits).to(device)
        else:
            raise NotImplementedError(f"Quantization method '{method}' not implemented.")

    @staticmethod
    def _create_uniform_map(symmetric=False, num_bits=4):
        if symmetric:
            negative = torch.linspace(-1, 0, 2 ** (num_bits - 1))
            positive = torch.linspace(0, 1, 2 ** (num_bits - 1))
            table = torch.cat([negative, positive[1:]])
        else:
            table = torch.linspace(-1, 1, 2**num_bits)
        return table

    @staticmethod
    def _create_normal_map(offset=0.9677083, symmetric=False, num_bits=2):
        try:
            from scipy.stats import norm
        except ImportError:
            raise ImportError("The required package 'scipy' is not installed for normal quantization.")
        
        variations = 2**num_bits
        if symmetric:
            v = norm.ppf(torch.linspace(1 - offset, offset, variations + 1)).tolist()
            values = torch.tensor([(0.5 * (v[i] + v[i+1])) for i in range(len(v) - 1)]).float()
        else: # Asymmetric, NF4-like approximation
            # Based on common NF4/NF2 values, normalized.
            # This is a simplified version. True NF4 uses specific quantile values.
            if num_bits == 4: 
                 # Example NF4 quantiles (not perfectly scaled/symmetric here for simplicity)
                example_nf4_like = [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
                                    0.0911, 0.1848, 0.2844, 0.3949, 0.5251, 0.6962, 1.0, 0.0] # 16th value for 2^4
                # Ensure 16 unique values for 2^4, often symmetric around 0
                # A more robust way for NF4:
                quantiles = torch.linspace(1e-2, 1.0 - 1e-2, variations) # Avoid exact 0 and 1 for ppf
                values = torch.tensor(norm.ppf(quantiles)).float()


            elif num_bits == 2:
                 values = torch.tensor([-1.0, -0.3333, 0.3333, 1.0]).float() # Common 2-bit values
            else: # General case
                 quantiles = torch.linspace(1e-2, 1.0 - 1e-2, variations)
                 values = torch.tensor(norm.ppf(quantiles)).float()
        
        if values.abs().max() > 1e-5: # Avoid division by zero if all values are tiny
            values = values / values.abs().max() # Normalize to [-1, 1]
        return values.sort().values


    def quantize_block(self, weight_2d):
        if not isinstance(weight_2d, torch.Tensor):
            raise TypeError("Input weight must be a PyTorch tensor.")
        if weight_2d.dim() != 2:
            raise ValueError(f"Input weight must be 2D. Got {weight_2d.dim()} dimensions.")
        if weight_2d.numel() == 0:
            return torch.empty(0, dtype=torch.uint8, device=self.device), \
                   torch.empty(0, dtype=weight_2d.dtype, device=self.device), \
                   weight_2d.shape

        original_shape = weight_2d.shape
        weight_flat = weight_2d.flatten()
        
        num_blocks = (weight_flat.numel() + self.block_size - 1) // self.block_size
        padded_size = num_blocks * self.block_size
        padding_needed = padded_size - weight_flat.numel()
        
        if padding_needed > 0:
            weight_flat_padded = F.pad(weight_flat, (0, padding_needed))
        else:
            weight_flat_padded = weight_flat

        weight_blocks_padded = weight_flat_padded.reshape(num_blocks, self.block_size)

        if self.method == "normal":
            block_abs_max = weight_blocks_padded.abs().max(dim=-1, keepdim=True)[0]
            scales = torch.where(block_abs_max == 0, torch.tensor(1.0, device=self.device, dtype=block_abs_max.dtype), block_abs_max)
        elif self.method == "uniform":
            block_min = weight_blocks_padded.min(dim=-1, keepdim=True)[0]
            block_max = weight_blocks_padded.max(dim=-1, keepdim=True)[0]
            scales = torch.maximum(block_min.abs(), block_max.abs())
            scales = torch.where(scales == 0, torch.tensor(1.0, device=self.device, dtype=scales.dtype), scales)
        else:
            raise NotImplementedError("Method not supported yet.")

        scaled_weights = weight_blocks_padded / scales
        
        # Ensure norm_lookup_table is on the same device
        self.norm_lookup_table = self.norm_lookup_table.to(scaled_weights.device)

        qweight_indices = torch.argmin(torch.abs(scaled_weights.unsqueeze(-1) - self.norm_lookup_table.view(1, 1, -1)), dim=-1)
        qweight_indices_flat = qweight_indices.flatten()

        if self.num_bits < 8:
            bits_per_byte = 8 // self.num_bits
            num_packed_elements = (qweight_indices_flat.numel() + bits_per_byte -1) // bits_per_byte
            packing_padding = num_packed_elements * bits_per_byte - qweight_indices_flat.numel()
            if packing_padding > 0:
                qweight_indices_flat = F.pad(qweight_indices_flat, (0, packing_padding))
            qweight_reshaped = qweight_indices_flat.reshape(-1, bits_per_byte)
            qweight_pack = torch.zeros((qweight_reshaped.shape[0], 1), dtype=torch.uint8, device=self.device)
            for i in range(bits_per_byte):
                qweight_pack[:, 0] |= qweight_reshaped[:, i].to(torch.uint8) << (i * self.num_bits)
        elif self.num_bits == 8:
            qweight_pack = qweight_indices_flat.to(torch.uint8).unsqueeze(-1)
        else:
            raise NotImplementedError("num_bits > 8 not directly supported for packing into uint8.")
        return qweight_pack, scales.squeeze(-1), original_shape


    def dequantize_block(self, qweight_pack, scales, target_shape):
        if qweight_pack.numel() == 0:
            return torch.empty(target_shape, dtype=scales.dtype, device=qweight_pack.device)

        self.norm_lookup_table = self.norm_lookup_table.to(qweight_pack.device) # Ensure on same device
        scales = scales.to(qweight_pack.device)


        if self.num_bits < 8:
            bits_per_byte = 8 // self.num_bits
            mask = (1 << self.num_bits) - 1
            unpacked_indices_list = []
            for i in range(bits_per_byte):
                shift = i * self.num_bits
                indices = (qweight_pack.squeeze(-1) >> shift) & mask
                unpacked_indices_list.append(indices.unsqueeze(-1)) # Keep as column
            
            qweight_indices_flat = torch.cat(unpacked_indices_list, dim=-1).view(-1).long()
        elif self.num_bits == 8:
            qweight_indices_flat = qweight_pack.squeeze(-1).long()
        else:
            raise NotImplementedError("num_bits > 8 not supported for unpacking from uint8.")
        
        dequantized_flat_padded_normalized = self.norm_lookup_table[qweight_indices_flat]
        
        num_blocks_scaled = scales.shape[0]
        block_size_used = dequantized_flat_padded_normalized.numel() // num_blocks_scaled
        
        dequantized_blocks_normalized = dequantized_flat_padded_normalized.reshape(num_blocks_scaled, block_size_used)
        
        scaled_blocks = dequantized_blocks_normalized * scales.unsqueeze(-1)
        
        scaled_flat_padded = scaled_blocks.view(-1)
        original_numel = target_shape.numel()
        dequantized_flat = scaled_flat_padded[:original_numel]
        
        return dequantized_flat.reshape(target_shape)

class TrueQuantizedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | str = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        quantization_bits: int = 4,
        reduced_rank: int = 16,
        num_iters: int = 1,
        quantization_method: str = "uniform",
        lora_A_kernel_size_from_orig: bool = True,
        lora_A_bias: bool = False,
        lora_B_bias: bool = False,
    ):
        super().__init__()
        self.in_channels_orig = in_channels
        self.out_channels_orig = out_channels

        self.kernel_size_orig = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride_orig = (stride, stride) if isinstance(stride, int) else stride
        self.padding_orig = padding
        self.dilation_orig = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups_orig = groups
        self.has_bias_orig = bias

        self.quantization_bits = quantization_bits
        self.reduced_rank = reduced_rank
        self.num_iters = num_iters
        self.quantization_method = quantization_method
        
        # Determine LoRA A's kernel, stride, padding, dilation
        lora_A_k_s_p_d = {
            "kernel_size": self.kernel_size_orig,
            "stride": self.stride_orig,
            "padding": self.padding_orig,
            "dilation": self.dilation_orig,
        } if lora_A_kernel_size_from_orig else {
            "kernel_size": (1,1), "stride": (1,1), "padding": 0, "dilation": (1,1),
        }

        self.lora_A = nn.Conv2d(
            in_channels=self.in_channels_orig, 
            out_channels=self.reduced_rank * self.groups_orig,
            groups=self.groups_orig,
            bias=lora_A_bias,
            **lora_A_k_s_p_d
        )
        self.lora_B = nn.Conv2d(
            in_channels=self.reduced_rank * self.groups_orig,
            out_channels=self.out_channels_orig,
            kernel_size=(1, 1),
            groups=self.groups_orig,
            bias=lora_B_bias
        )

        if self.has_bias_orig:
            self.bias = nn.Parameter(torch.zeros(self.out_channels_orig))
        else:
            self.register_parameter('bias', None)

        self.quantizer = _BlockQuantizer(
            num_bits=quantization_bits, 
            device=compute_device,
            method=quantization_method,
        )

    def _low_rank_decomposition(self, weight_2d: torch.Tensor, reduced_rank: int):
        U, S, Vh = torch.linalg.svd(weight_2d.to(torch.float32), full_matrices=False) # SVD on float32
        L_factor = U[:, :reduced_rank] @ torch.diag(S[:reduced_rank].sqrt())
        R_factor = torch.diag(S[:reduced_rank].sqrt()) @ Vh[:reduced_rank, :]
        return L_factor.to(weight_2d.dtype), R_factor.to(weight_2d.dtype)

    def _loftq(self, weight_2d: torch.Tensor, reduced_rank: int, num_iter: int):
        # Simplified LoftQ for T=1 (num_iter=1)
        # 1. Quantize W to get Q_W.
        # 2. Perform SVD on W - Q_W to get L, R.
        current_dtype = weight_2d.dtype
        
        # Q_1 = q_N(W)
        qpack_base, scales_base, _ = self.quantizer.quantize_block(weight_2d)
        dequantized_base_2d = self.quantizer.dequantize_block(qpack_base, scales_base, weight_2d.shape)

        # Residual for SVD: W - Q_1
        residual_2d = weight_2d - dequantized_base_2d
        
        # A_1, B_1 = SVD(W - Q_1)
        # Here, L_factor corresponds to L (or A in AB^T), R_factor to R (or B^T in AB^T)
        lora_L_2d, lora_R_2d = self._low_rank_decomposition(residual_2d, reduced_rank)
        
        return qpack_base, scales_base, lora_L_2d, lora_R_2d

    def quantize(self, original_conv_layer: nn.Conv2d):
        W_initial_4d = original_conv_layer.weight.data.clone()
        C_out, C_in_div_groups, kH, kW = W_initial_4d.shape
        
        # Reshape 4D kernel to 2D: (C_out, C_in_div_groups * kH * kW)
        W_initial_2d = W_initial_4d.reshape(C_out, -1)

        qweight_pack_2d, weight_scales_2d, lora_L_2d, lora_R_2d = self._loftq(
            W_initial_2d, 
            self.reduced_rank, # This is the rank for the SVD of the (C_out, C_in_eff) matrix
            self.num_iters
        )
        # lora_L_2d is (C_out, rank_svd)
        # lora_R_2d is (rank_svd, C_in_div_groups * kH * kW)

        self.register_buffer('qweight', qweight_pack_2d)
        self.register_buffer('weight_scales', weight_scales_2d)
        self.register_buffer('kernel_shape_orig_buf', torch.tensor(W_initial_4d.shape, device=compute_device))
        self.register_buffer('reshaped_2d_shape_buf', torch.tensor(W_initial_2d.shape, device=compute_device))


        # lora_A.weight: (rank_A_out_total, C_in_div_groups, kH_A, kW_A)
        # rank_A_out_total = self.reduced_rank * self.groups_orig
        # lora_R_2d needs to be (rank_A_out_total, C_in_div_groups * kH * kW)
        # The SVD rank (self.reduced_rank) must match lora_B's input rank per group,
        # or lora_A's output rank per group.
        # If SVD rank is `r`, then lora_R_2d is (r, C_in_eff).
        # lora_A has out_channels = self.reduced_rank * self.groups_orig.
        # So, the SVD rank should conceptually be self.reduced_rank * self.groups_orig.
        # Let's assume self.reduced_rank in __init__ is the rank for the SVD.
        
        # lora_A.weight shape: (self.reduced_rank * self.groups_orig, C_in_div_groups, kH_A, kW_A)
        # lora_R_2d shape: (self.reduced_rank, C_in_div_groups * kH * kW)
        # This means lora_R_2d must be tiled or interpreted for groups if SVD rank != lora_A.out_channels
        
        # Assuming SVD rank = self.reduced_rank (total rank for the matrix factorization)
        # And lora_A.out_channels = self.reduced_rank * self.groups_orig
        # This implies the SVD should have been done with rank = self.reduced_rank * self.groups_orig
        # For now, let's assume self.reduced_rank is the total rank for SVD,
        # and lora_A.out_channels = self.reduced_rank (if groups=1) or rank*groups.
        # The current lora_A definition has out_channels = self.reduced_rank * self.groups_orig.
        # So, lora_R_2d should be (self.reduced_rank * self.groups_orig, C_in_div_groups * kH_A * kW_A)
        # This means the rank passed to SVD should be self.reduced_rank * self.groups_orig.
        
        # Let's redefine rank_for_svd for clarity
        rank_for_svd = self.lora_A.out_channels # self.reduced_rank * self.groups_orig
        
        # Re-run _loftq if rank definition changed (this is a bit of a patch)
        if W_initial_2d.shape[0] < rank_for_svd or W_initial_2d.shape[1] < rank_for_svd:
             actual_rank_for_svd = min(W_initial_2d.shape[0], W_initial_2d.shape[1], rank_for_svd)
        else:
            actual_rank_for_svd = rank_for_svd

        if actual_rank_for_svd != self.reduced_rank: # If our assumption for SVD rank was different
            qweight_pack_2d, weight_scales_2d, lora_L_2d, lora_R_2d = self._loftq(
                W_initial_2d, actual_rank_for_svd, self.num_iters
            )
        
        # lora_L_2d: (C_out, actual_rank_for_svd)
        # lora_R_2d: (actual_rank_for_svd, C_in_div_groups * kH * kW)

        # lora_A.weight: (lora_A_out_channels, C_in_div_groups, kH_A, kW_A)
        # lora_A_out_channels = self.reduced_rank * self.groups_orig (which is actual_rank_for_svd)
        kH_A, kW_A = self.lora_A.kernel_size
        self.lora_A.weight.data = lora_R_2d.reshape(
            actual_rank_for_svd, C_in_div_groups, kH_A, kW_A
        ).to(self.lora_A.weight.dtype)

        # lora_B.weight: (C_out, lora_B_in_channels_div_groups, 1, 1)
        # lora_B_in_channels_div_groups = (self.reduced_rank * self.groups_orig) / self.groups_orig = self.reduced_rank
        # So, lora_B.weight is (C_out, self.reduced_rank, 1, 1)
        # lora_L_2d is (C_out, actual_rank_for_svd)
        # We need to ensure actual_rank_for_svd matches lora_B's input channel structure.
        # lora_B.in_channels = self.reduced_rank * self.groups_orig = actual_rank_for_svd
        # lora_B.weight shape: (self.out_channels_orig, self.reduced_rank_per_group_for_B, 1, 1)
        # where reduced_rank_per_group_for_B = lora_B.in_channels / lora_B.groups
        # = (self.reduced_rank * self.groups_orig) / self.groups_orig = self.reduced_rank
        
        self.lora_B.weight.data = lora_L_2d.reshape(
            self.out_channels_orig, self.reduced_rank, 1, 1 # Assuming self.reduced_rank is rank per group for B's input
        ).to(self.lora_B.weight.dtype)


        if self.has_bias_orig and original_conv_layer.bias is not None:
            self.bias.data = original_conv_layer.bias.data.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reshaped_2d_kernel_shape = tuple(self.reshaped_2d_shape_buf.tolist())
        
        dequantized_kernel_2d = self.quantizer.dequantize_block(
            self.qweight, 
            self.weight_scales, 
            reshaped_2d_kernel_shape
        )
        
        original_4d_kernel_shape = tuple(self.kernel_shape_orig_buf.tolist())
        dequantized_kernel_4d = dequantized_kernel_2d.reshape(original_4d_kernel_shape)
        
        base_output = F.conv2d(
            x, 
            dequantized_kernel_4d, 
            self.bias, 
            self.stride_orig, 
            self.padding_orig, 
            self.dilation_orig, 
            self.groups_orig
        )
        
        lora_output = self.lora_B(self.lora_A(x))
        
        return base_output + lora_output

