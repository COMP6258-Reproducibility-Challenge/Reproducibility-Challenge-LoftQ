import torch
import torch.nn as nn
import torch.nn.functional as F
import copy # For deep copying the model

# Assuming loftq_cnn.py and model_utils.py are in the same directory or accessible in PYTHONPATH
# If they are in 'custom_loftq' and this script is also in 'custom_loftq':
from loftq_cnn import TrueQuantizedConv2d, compute_device
from model_utils import convert_true_quant_conv_layer

# Define a CNN model with two sequential convolutional layers
class SequentialConvCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SequentialConvCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1) # Second conv layer
        self.relu2 = nn.ReLU()

        # Calculate the input size for the fully connected layer
        # Assuming input image size 3x32x32:
        # After conv1 (3,3,1 padding 1 on 32x32): output size 16x32x32
        # After conv2 (3,3,1 padding 1 on 32x32): output size 32x32x32
        self.fc_input_features = 32 * 32 * 32
        self.fc = nn.Linear(self.fc_input_features, num_classes)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x

# Define mock model arguments (can be imported or redefined)
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
        # For conv layers, output is a tensor. For some layers, it might be a tuple.
        if isinstance(output, torch.Tensor):
            activation_outputs[name] = output.detach()
        else: # If output is a tuple (e.g. some RNNs, or custom layers)
            activation_outputs[name] = output[0].detach() # Assuming the first element is the main output
    return hook

if __name__ == "__main__":
    # --- Setup ---
    torch.manual_seed(0) # For reproducible dummy input
    original_cnn = SequentialConvCNN(num_classes=10).to(compute_device)
    # Create a deep copy for conversion to keep original_cnn pristine
    converted_cnn = copy.deepcopy(original_cnn)

    # Configure model arguments for quantization
    # Rank 8 for conv1 (16 out_channels), Rank 16 for conv2 (32 out_channels) could be an option
    # For simplicity, using a single rank here. Adjust if needed.
    model_args = MockModelArgs(int_bit=4, reduced_rank=8, num_iter=1, quant_method="normal")

    print("Original SequentialConvCNN Model Structure:")
    print(original_cnn)
    print("-" * 50)

    # --- Conversion ---
    # The convert_true_quant_conv_layer will modify converted_cnn in-place
    converted_cnn = convert_true_quant_conv_layer(converted_cnn, model_args)
    print("\nConverted SequentialConvCNN Model Structure:")
    print(converted_cnn)
    print("-" * 50)

    # --- Prepare dummy input ---
    dummy_input = torch.randn(2, 3, 32, 32).to(compute_device) # Batch size 2

    # --- Register hooks to capture conv1 and conv2 outputs ---
    # Hooks for original model
    original_cnn.conv1.register_forward_hook(get_activation('original_conv1'))
    original_cnn.conv2.register_forward_hook(get_activation('original_conv2'))

    # Hooks for converted model
    converted_cnn.conv1.register_forward_hook(get_activation('converted_conv1')) # This is now TrueQuantizedConv2d
    converted_cnn.conv2.register_forward_hook(get_activation('converted_conv2')) # This is also TrueQuantizedConv2d

    # --- Perform forward passes ---
    print("\nPerforming forward passes...")
    # Original model
    original_full_output = original_cnn(dummy_input)
    original_conv1_hook_output = activation_outputs.get('original_conv1')
    original_conv2_hook_output = activation_outputs.get('original_conv2')

    # Converted model
    converted_full_output = converted_cnn(dummy_input)
    converted_conv1_hook_output = activation_outputs.get('converted_conv1')
    converted_conv2_hook_output = activation_outputs.get('converted_conv2')

    # --- Calculate MSE for conv1 layer outputs ---
    print("\n--- MSE for conv1 ---")
    if original_conv1_hook_output is not None and converted_conv1_hook_output is not None:
        if original_conv1_hook_output.shape == converted_conv1_hook_output.shape:
            mse_conv1 = F.mse_loss(original_conv1_hook_output, converted_conv1_hook_output)
            print(f"Shape of original_conv1_output: {original_conv1_hook_output.shape}")
            print(f"Shape of converted_conv1_output: {converted_conv1_hook_output.shape}")
            print(f"MSE between original conv1 and TrueQuantizedConv2d (conv1) outputs: {mse_conv1.item():.8f}")
        else:
            print("Error: Shapes of conv1 outputs do not match.")
    else:
        print("Error: Could not capture conv1 outputs. Check hook registration or model structure.")

    # --- Calculate MSE for conv2 layer outputs ---
    print("\n--- MSE for conv2 ---")
    if original_conv2_hook_output is not None and converted_conv2_hook_output is not None:
        if original_conv2_hook_output.shape == converted_conv2_hook_output.shape:
            mse_conv2 = F.mse_loss(original_conv2_hook_output, converted_conv2_hook_output)
            print(f"Shape of original_conv2_output: {original_conv2_hook_output.shape}")
            print(f"Shape of converted_conv2_output: {converted_conv2_hook_output.shape}")
            print(f"MSE between original conv2 and TrueQuantizedConv2d (conv2) outputs: {mse_conv2.item():.8f}")
        else:
            print("Error: Shapes of conv2 outputs do not match.")
    else:
        print("Error: Could not capture conv2 outputs. Check hook registration or model structure.")

    # --- Calculate MSE for full model outputs ---
    print("\n--- MSE for Full Model ---")
    if original_full_output.shape == converted_full_output.shape:
        mse_full_model = F.mse_loss(original_full_output, converted_full_output)
        print(f"Shape of original_full_output: {original_full_output.shape}")
        print(f"Shape of converted_full_output: {converted_full_output.shape}")
        print(f"MSE between original full model and converted full model outputs: {mse_full_model.item():.8f}")
    else:
        print("Error: Shapes of full model outputs do not match.")

    print("-" * 50)
    print("Script finished.")
