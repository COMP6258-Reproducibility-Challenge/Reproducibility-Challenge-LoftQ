import torch
import torch.nn as nn
import copy # For deep copying the model

# Assuming loftq_cnn.py and model_utils.py are in the same directory or accessible in PYTHONPATH
from loftq_cnn import TrueQuantizedConv2d, compute_device, _BlockQuantizer # Import _BlockQuantizer if needed for direct dequant
from model_utils import convert_true_quant_conv_layer

# Define a very simple CNN model (same as in test_cnn_conversion.py)
class MinimalCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MinimalCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.fc_input_features = 16 * 32 * 32 # Assuming 3x32x32 input
        self.fc = nn.Linear(self.fc_input_features, num_classes)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x

# Define mock model arguments
class MockModelArgs:
    def __init__(self, int_bit=4, reduced_rank=16, num_iter=1, quant_method="uniform"):
        self.int_bit = int_bit
        self.reduced_rank = reduced_rank
        self.num_iter = num_iter
        self.quant_method = quant_method

if __name__ == "__main__":
    # --- Setup ---
    torch.manual_seed(0) # For reproducible weights if layers are re-initialized
    original_cnn = MinimalCNN(num_classes=10).to(compute_device)
    converted_cnn = copy.deepcopy(original_cnn) # Create a deep copy for conversion

    model_args = MockModelArgs(int_bit=4, reduced_rank=8, num_iter=1, quant_method="normal")

    print("Original Minimal Model Structure:")
    print(original_cnn)
    print("-" * 50)

    # --- Conversion ---
    converted_cnn = convert_true_quant_conv_layer(converted_cnn, model_args)
    print("\nConverted Minimal Model Structure:")
    print(converted_cnn) # This will show the TrueQuantizedConv2d layer
    print("-" * 50)

    # --- Accessing Kernels ---
    original_kernel = original_cnn.conv1.weight.data.clone()

    # Accessing the TrueQuantizedConv2d layer
    true_quant_conv_layer = converted_cnn.conv1
    if not isinstance(true_quant_conv_layer, TrueQuantizedConv2d):
        print("Error: conv1 layer in converted_cnn is not a TrueQuantizedConv2d instance.")
        exit()

    # Retrieve the necessary components from TrueQuantizedConv2d
    # These are buffers, so access them directly
    q_weight_packed = true_quant_conv_layer.qweight
    weight_scales = true_quant_conv_layer.weight_scales
    # The shape used for dequantization should be the 2D shape the quantizer saw
    reshaped_2d_kernel_shape_tensor = true_quant_conv_layer.reshaped_2d_shape_buf
    reshaped_2d_kernel_shape_tuple = tuple(reshaped_2d_kernel_shape_tensor.tolist())

    # The original 4D shape for final reshaping
    original_4d_kernel_shape_tensor = true_quant_conv_layer.kernel_shape_orig_buf
    original_4d_kernel_shape_tuple = tuple(original_4d_kernel_shape_tensor.tolist())


    # Dequantize the kernel using the quantizer instance from the layer
    # The _BlockQuantizer instance is self.quantizer inside TrueQuantizedConv2d
    dequantized_kernel_2d = true_quant_conv_layer.quantizer.dequantize_block(
        q_weight_packed,
        weight_scales,
        reshaped_2d_kernel_shape_tuple # Pass the 2D shape it was quantized with
    )
    
    # Reshape the dequantized 2D kernel back to its original 4D shape
    dequantized_kernel_4d = dequantized_kernel_2d.reshape(original_4d_kernel_shape_tuple)

    print("\n--- Kernel Inspection ---")
    print(f"Original Kernel Shape: {original_kernel.shape}")
    print(f"Dequantized Kernel Shape: {dequantized_kernel_4d.shape}")

    # --- Visual Comparison (Print a small slice) ---
    # Example: Print the 3x3 kernel for the first output channel, first input channel
    print("\nOriginal Kernel (slice [0, 0, :, :]):")
    print(original_kernel[0, 0, :, :])

    print("\nDequantized Kernel (slice [0, 0, :, :]):")
    print(dequantized_kernel_4d[0, 0, :, :])
    
    # --- Inspecting unique values in the dequantized kernel ---
    # Flatten the dequantized kernel to easily find unique values
    unique_values_in_dequantized_kernel = torch.unique(dequantized_kernel_4d.flatten())
    print(f"\nNumber of unique values in the dequantized kernel: {len(unique_values_in_dequantized_kernel)}")
    
    # Print a few unique values (can be many if scales are diverse)
    print("Some unique values from the dequantized kernel (up to 20):")
    if len(unique_values_in_dequantized_kernel) > 20:
        print(unique_values_in_dequantized_kernel[:20])
        print("...")
    else:
        print(unique_values_in_dequantized_kernel)

    # For 4-bit quantization (2^4 = 16 levels in norm_lookup_table)
    # The number of unique values will be roughly num_lookup_levels * num_blocks (if scales differ per block)
    # or exactly num_lookup_levels if all blocks share the same scale (less likely for 'normal' method).
    # If you want to see the effect of the norm_lookup_table more directly,
    # you could inspect `true_quant_conv_layer.quantizer.norm_lookup_table`
    # and relate it to the `scaled_weights` within the dequantization logic.
    
    print(f"\nQuantizer's norm_lookup_table (first {min(16, len(true_quant_conv_layer.quantizer.norm_lookup_table))} values):")
    print(true_quant_conv_layer.quantizer.norm_lookup_table[:min(16, len(true_quant_conv_layer.quantizer.norm_lookup_table))])


    print("-" * 50)
    print("Script finished.")
