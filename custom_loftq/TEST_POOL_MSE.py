import torch
import torch.nn as nn
import torch.nn.functional as F
import copy # For deep copying the model

# Assuming loftq_cnn.py and model_utils.py are in the same directory or accessible in PYTHONPATH
from loftq_cnn import TrueQuantizedConv2d, compute_device # Ensure TrueQuantizedConv2d is imported
from model_utils import convert_true_quant_conv_layer

# Define a CNN model with a Convolutional layer followed by Max Pooling
class ConvPoolCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvPoolCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Max pooling layer

        # Calculate the input size for the fully connected layer
        # Assuming input image size 3x32x32:
        # After conv1 (3,3,1 padding 1 on 32x32): output size 16x32x32
        # After pool1 (2x2 kernel, stride 2 on 32x32): output size 16x16x16
        self.fc_input_features = 16 * 16 * 16
        self.fc = nn.Linear(self.fc_input_features, num_classes)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
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
        if isinstance(output, torch.Tensor):
            activation_outputs[name] = output.detach()
        else:
            activation_outputs[name] = output[0].detach()
    return hook

if __name__ == "__main__":
    # --- Setup ---
    torch.manual_seed(0)
    original_cnn = ConvPoolCNN(num_classes=10).to(compute_device)
    converted_cnn = copy.deepcopy(original_cnn)

    model_args = MockModelArgs(int_bit=4, reduced_rank=8, num_iter=1, quant_method="normal")

    print("Original ConvPoolCNN Model Structure:")
    print(original_cnn)
    print("-" * 50)

    # --- Conversion ---
    converted_cnn = convert_true_quant_conv_layer(converted_cnn, model_args)
    print("\nConverted ConvPoolCNN Model Structure:")
    print(converted_cnn)
    print("-" * 50)

    # --- Prepare dummy input ---
    dummy_input = torch.randn(2, 3, 32, 32).to(compute_device)

    # --- Register hooks ---
    # Hooks for original model
    original_cnn.conv1.register_forward_hook(get_activation('original_conv1'))
    original_cnn.pool1.register_forward_hook(get_activation('original_pool1'))

    # Hooks for converted model
    converted_cnn.conv1.register_forward_hook(get_activation('converted_conv1')) # This is TrueQuantizedConv2d
    converted_cnn.pool1.register_forward_hook(get_activation('converted_pool1')) # pool1 is not converted, but its input changes

    # --- Perform forward passes ---
    print("\nPerforming forward passes...")
    original_full_output = original_cnn(dummy_input)
    original_conv1_hook_output = activation_outputs.get('original_conv1')
    original_pool1_hook_output = activation_outputs.get('original_pool1')

    converted_full_output = converted_cnn(dummy_input)
    converted_conv1_hook_output = activation_outputs.get('converted_conv1')
    converted_pool1_hook_output = activation_outputs.get('converted_pool1')

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
        print("Error: Could not capture conv1 outputs.")

    # --- Calculate MSE for pool1 layer outputs ---
    print("\n--- MSE for pool1 ---")
    if original_pool1_hook_output is not None and converted_pool1_hook_output is not None:
        if original_pool1_hook_output.shape == converted_pool1_hook_output.shape:
            mse_pool1 = F.mse_loss(original_pool1_hook_output, converted_pool1_hook_output)
            print(f"Shape of original_pool1_output: {original_pool1_hook_output.shape}")
            print(f"Shape of converted_pool1_output: {converted_pool1_hook_output.shape}")
            print(f"MSE between original pool1 and converted model's pool1 outputs: {mse_pool1.item():.8f}")
        else:
            print("Error: Shapes of pool1 outputs do not match.")
    else:
        print("Error: Could not capture pool1 outputs.")

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
