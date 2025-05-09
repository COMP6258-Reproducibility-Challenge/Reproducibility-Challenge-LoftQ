import torch
import torch.nn as nn

# Assuming TrueQuantizedConv2d and convert_true_quant_conv_layer are accessible
# For example, if they are in custom_loftq.loftq_cnn and custom_loftq.model_utils_extended
# from custom_loftq.loftq_cnn import TrueQuantizedConv2d 
# from custom_loftq.model_utils_extended import convert_true_quant_conv_layer

# If running this script directly and the files are in the same directory (for testing):
from loftq_cnn import TrueQuantizedConv2d, compute_device # Make sure compute_device is imported
from model_utils_extended import convert_true_quant_conv_layer


# Define a simple CNN model for testing
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, groups=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, groups=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_grouped = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, groups=2) # Grouped conv
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(64 * 8 * 8, num_classes) # Assuming input image size leads to 8x8 feature map

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3_grouped(x))
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x

# Define mock model arguments (similar to your arguments.py)
class MockModelArgs:
    def __init__(self, int_bit=4, reduced_rank=16, num_iter=1, quant_method="uniform"):
        self.int_bit = int_bit
        self.reduced_rank = reduced_rank
        self.num_iter = num_iter
        self.quant_method = quant_method
        # Add any other args TrueQuantizedConv2d or convert_true_quant_conv_layer might expect

if __name__ == "__main__":
    # Instantiate the simple CNN
    original_model = SimpleCNN(num_classes=10).to(compute_device)
    print("Original Model Structure:")
    print(original_model)
    print("-" * 50)

    # Create mock model arguments
    model_args = MockModelArgs(int_bit=4, reduced_rank=8, num_iter=1, quant_method="normal")
                               # Reduced rank for smaller test kernels

    # Convert the model
    # Make sure TrueQuantizedConv2d is correctly imported in model_utils_extended.py
    converted_model = convert_true_quant_conv_layer(original_model, model_args)
    
    print("\nConverted Model Structure:")
    print(converted_model)
    print("-" * 50)

    # Test forward pass with a dummy input
    # Assuming input images are 3x32x32 (channels, height, width)
    # After conv1 (16,32,32), pool1 (16,16,16)
    # After conv2 (32,16,16), pool2 (32,8,8)
    # After conv3_grouped (64,8,8)
    # FC input: 64 * 8 * 8
    dummy_input = torch.randn(2, 3, 32, 32).to(compute_device) # Batch size 2
    
    print("\nTesting forward pass of the original model...")
    try:
        original_output = original_model(dummy_input) # This will be the converted model if conversion is in-place
                                                      # Let's re-init for a true original test
        original_model_for_test = SimpleCNN(num_classes=10).to(compute_device)
        original_output = original_model_for_test(dummy_input)
        print("Original model forward pass successful. Output shape:", original_output.shape)
    except Exception as e:
        print("Error during original model forward pass:", e)
        import traceback
        traceback.print_exc()

    print("\nTesting forward pass of the converted model...")
    try:
        converted_output = converted_model(dummy_input)
        print("Converted model forward pass successful. Output shape:", converted_output.shape)
    except Exception as e:
        print("Error during converted model forward pass:", e)
        import traceback
        traceback.print_exc()

    print("-" * 50)
    print("Script finished.")

