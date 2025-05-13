import torch
import torch.nn as nn

from loftq_cnn import TrueQuantizedConv2d, compute_device
from model_utils import convert_true_quant_conv_layer


class MinimalCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MinimalCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.fc_input_features = 16 * 32 * 32  # Based on 3x32x32 input and same padding
        self.fc = nn.Linear(self.fc_input_features, num_classes)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MockModelArgs:
    def __init__(self, int_bit=4, reduced_rank=16, num_iter=1, quant_method="uniform"):
        self.int_bit = int_bit
        self.reduced_rank = reduced_rank
        self.num_iter = num_iter
        self.quant_method = quant_method


if __name__ == "__main__":
    original_model = MinimalCNN(num_classes=10).to(compute_device)
    print("Original Minimal Model Structure:")
    print(original_model)
    print("-" * 50)

    model_args = MockModelArgs(int_bit=4, reduced_rank=8, num_iter=1, quant_method="normal")
    converted_model = convert_true_quant_conv_layer(original_model, model_args)

    print("\nConverted Minimal Model Structure:")
    print(converted_model)
    print("-" * 50)

    dummy_input = torch.randn(2, 3, 32, 32).to(compute_device)

    print("\nTesting forward pass of the original minimal model...")
    try:
        original_model_for_test = MinimalCNN(num_classes=10).to(compute_device)
        original_output = original_model_for_test(dummy_input)
        print("Original minimal model forward pass successful. Output shape:", original_output.shape)
    except Exception as e:
        print("Error during original minimal model forward pass:", e)
        import traceback
        traceback.print_exc()

    print("\nTesting forward pass of the converted minimal model...")
    try:
        converted_output = converted_model(dummy_input)
        print("Converted minimal model forward pass successful. Output shape:", converted_output.shape)
    except Exception as e:
        print("Error during converted minimal model forward pass:", e)
        import traceback
        traceback.print_exc()

    print("-" * 50)
    print("Script finished.")
