import torch
import torch.nn as nn
import torch.nn.functional as F
import copy # For deep copying the model

# Assuming loftq_cnn.py and model_utils.py are in the same directory or accessible in PYTHONPATH
from loftq_cnn import TrueQuantizedConv2d, compute_device
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

# --- Hook for capturing intermediate layer output ---
activation_outputs = {}
def get_activation(name):
    def hook(model, input, output):
        activation_outputs[name] = output.detach()
    return hook

if __name__ == "__main__":
    # --- Setup ---
    torch.manual_seed(0) # For reproducible dummy input
    original_cnn = MinimalCNN(num_classes=10).to(compute_device)
    converted_cnn = copy.deepcopy(original_cnn) # Create a deep copy for conversion

    model_args = MockModelArgs(int_bit=4, reduced_rank=8, num_iter=1, quant_method="normal")

    print("Original Minimal Model Structure:")
    print(original_cnn)
    print("-" * 50)

    # --- Conversion ---
    converted_cnn = convert_true_quant_conv_layer(converted_cnn, model_args)
    print("\nConverted Minimal Model Structure:")
    print(converted_cnn)
    print("-" * 50)

    # --- Prepare dummy input ---
    dummy_input = torch.randn(2, 3, 32, 32).to(compute_device) # Batch size 2

    # --- Register hooks to capture conv1 output ---
    # Hook for original model's conv1
    original_cnn.conv1.register_forward_hook(get_activation('original_conv1'))
    # Hook for converted model's conv1 (which is now TrueQuantizedConv2d)
    converted_cnn.conv1.register_forward_hook(get_activation('converted_conv1'))

    # --- Perform forward passes ---
    print("\nPerforming forward passes...")
    # Original model
    original_full_output = original_cnn(dummy_input)
    original_conv1_output = activation_outputs.get('original_conv1')

    # Converted model
    converted_full_output = converted_cnn(dummy_input)
    converted_conv1_output = activation_outputs.get('converted_conv1')

    if original_conv1_output is None or converted_conv1_output is None:
        print("Error: Could not capture conv1 outputs. Check hook registration.")
    else:
        print(f"Shape of original_conv1_output: {original_conv1_output.shape}")
        print(f"Shape of converted_conv1_output: {converted_conv1_output.shape}")

        # --- Calculate MSE for conv1 layer outputs ---
        if original_conv1_output.shape == converted_conv1_output.shape:
            mse_conv1 = F.mse_loss(original_conv1_output, converted_conv1_output)
            print(f"\nMSE between original conv1 and TrueQuantizedConv2d (conv1) outputs: {mse_conv1.item():.8f}")
            
            # You might also want to compare with the output of just the base dequantized layer
            # of TrueQuantizedConv2d, without the LoRA part, to see the quantization error alone.
            # This would require modifying TrueQuantizedConv2d to optionally return that.
            # For now, we compare the full TrueQuantizedConv2d output (base + LoRA).
            
            # Example: To get just the base output from TrueQuantizedConv2d (requires modification to TrueQuantizedConv2d)
            # if hasattr(converted_cnn.conv1, 'get_base_output'):
            #     base_only_output = converted_cnn.conv1.get_base_output(dummy_input) # Hypothetical method
            #     mse_quant_error_only = F.mse_loss(original_conv1_output, base_only_output)
            #     print(f"MSE for quantization error only (original vs. dequantized base): {mse_quant_error_only.item():.8f}")

        else:
            print("Error: Shapes of conv1 outputs do not match, cannot calculate MSE.")
            print(f"Original conv1 output shape: {original_conv1_output.shape}")
            print(f"Converted conv1 output shape: {converted_conv1_output.shape}")

    # --- Calculate MSE for full model outputs ---
    if original_full_output.shape == converted_full_output.shape:
        mse_full_model = F.mse_loss(original_full_output, converted_full_output)
        print(f"\nMSE between original full model and converted full model outputs: {mse_full_model.item():.8f}")
    else:
        print("Error: Shapes of full model outputs do not match, cannot calculate MSE for full model.")

    print("-" * 50)
    print("Script finished.")