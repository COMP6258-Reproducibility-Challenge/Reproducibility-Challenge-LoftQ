import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import copy
import csv
import os
import time

# Assuming your LoftQ modules are in the current directory or accessible in PYTHONPATH
from loftq_cnn import TrueQuantizedConv2d, compute_device
from model_utils import convert_true_quant_conv_layer
from loftq import TrueQuantizedLinear
from model_utils import convert_linear_layer

# --- 1. Define the "SimpleCNN" Model (from 5_1_cnn.py) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_rate = 0.2
        self.fc1_input_features = 32 * 12 * 12
        self.fc1 = nn.Linear(self.fc1_input_features, 128)
        self.relu2 = nn.ReLU()
        self.fc2_final = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = x.view(-1, self.fc1_input_features)
        x = self.relu2(self.fc1(x))
        x = self.fc2_final(x)
        return x

# --- Mock Model Arguments for LoftQ ---
class MockModelArgs:
    def __init__(self, int_bit=4, reduced_rank=16, num_iter=1, quant_method="normal", true_quantization=True):
        self.int_bit = int_bit
        self.reduced_rank = reduced_rank
        self.num_iter = num_iter
        self.quant_method = quant_method
        self.true_quantization = true_quantization

# --- 2. Data Loading and Training (MNIST) ---
def get_mnist_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader

def train_model(model, trainloader, criterion, optimizer, epochs=3, device=compute_device, model_path='simple_cnn_mnist_original.pth'):
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded pre-trained original model weights from {model_path}.")
        return model

    model.train()
    print(f"Starting training on {device} for {epochs} epochs...")
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'[Epoch {epoch + 1}, Batch {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
    print('Finished Training')
    torch.save(model.state_dict(), model_path)
    print(f"Saved trained model weights to {model_path}.")
    return model

# --- 3. Quantization Function ---
def quantize_simple_cnn_model(model_to_quantize, model_args, quantize_final_fc_layer=True):
    # print(f"Quantizing with args: bits={model_args.int_bit}, rank={model_args.reduced_rank}, method='{model_args.quant_method}', final_fc_quant={quantize_final_fc_layer}")
    
    linear_target_modules = ['fc1']
    if quantize_final_fc_layer:
        linear_target_modules.append('fc2_final')
    
    linear_excluded_modules = []
    if not quantize_final_fc_layer:
        linear_excluded_modules.append('fc2_final')

    model_to_quantize = convert_true_quant_conv_layer(model_to_quantize, model_args)
    model_to_quantize = convert_linear_layer(
        model_to_quantize,
        quantization_bits=model_args.int_bit,
        rank=model_args.reduced_rank,
        num_iters=model_args.num_iter,
        quantization_method=model_args.quant_method,
        target_modules=linear_target_modules,
        excluded_modules=linear_excluded_modules,
        true_quantization=model_args.true_quantization
    )
    return model_to_quantize

# --- 5. Evaluation ---
@torch.no_grad() # Decorator for no_grad context
def evaluate_model(model, testloader, device=compute_device):
    model.eval()
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# --- 6. Main Experiment Flow ---
if __name__ == '__main__':
    device = compute_device
    trainloader, testloader = get_mnist_loaders()
    
    original_model_path = 'simple_cnn_mnist_original.pth'
    original_model_instance = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(original_model_instance.parameters(), lr=0.001)

    # Train or load the original model
    original_model_instance = train_model(original_model_instance, trainloader, criterion, optimizer, epochs=3, device=device, model_path=original_model_path)
    
    print("\n--- Evaluating Original Model ---")
    original_accuracy = evaluate_model(original_model_instance, testloader, device=device)
    print(f"Original Model Accuracy: {original_accuracy:.2f} %")

    # --- CSV Setup ---
    csv_file_path = 'loftq_cnn_experiment_results.csv'
    csv_headers = ['int_bit', 'reduced_rank', 'quant_method', 'final_layer_quantized', 'accuracy', 'time_taken_quant_eval_s']
    
    # Write headers if file doesn't exist
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

    # --- Parameter Ranges for Sweep ---
    bits_to_test = [2, 4, 8]
    ranks_to_test = [8, 16] # Ensure rank is not too large for fc2_final (out_features=10)
    quant_methods_to_test = ["uniform", "normal"]
    final_layer_quant_options = [True, False]

    print("\n\n--- Starting Quantization Parameter Sweep ---")

    for bits in bits_to_test:
        for rank in ranks_to_test:
            # Adjust rank if it's too large for the smallest dimension of a layer being quantized
            # For fc2_final (128 -> 10), max rank is 10.
            # For fc1 (4608 -> 128), max rank is 128.
            # For conv1 (1 -> 32, kernel 5x5 -> 25 effective inputs per output channel), rank is relative to 32 and 25.
            # Let's cap rank for fc2_final if it's being quantized.
            # A more robust way would be to check layer dimensions before setting rank in MockModelArgs for each layer.
            # For simplicity here, we use the loop 'rank' but are mindful of fc2_final.
            current_rank_fc2 = min(rank, 10) # Max rank for fc2_final is 10
            current_rank_fc1 = min(rank, 128)
            current_rank_conv1 = min(rank, 25) # Approximate for conv1

            for method in quant_methods_to_test:
                for final_quant_status in final_layer_quant_options:
                    start_time = time.time()
                    print("-" * 60)
                    print(f"Testing: Bits={bits}, Rank={rank} (fc1:{current_rank_fc1}, fc2:{current_rank_fc2 if final_quant_status else 'N/A'}, conv1:{current_rank_conv1}), Method='{method}', FinalFCQuant={final_quant_status}")

                    # Use the appropriate rank for MockModelArgs.
                    # This simplified MockModelArgs uses one rank for all. A more advanced setup
                    # might pass different ranks per layer. For now, we'll use the loop 'rank'.
                    # If a layer is smaller than 'rank', your LoftQ should handle it (e.g., by using min(rank, dim)).
                    loftq_args = MockModelArgs(
                        int_bit=bits,
                        reduced_rank=rank, # The convert_ functions should handle if rank > layer_dim
                        quant_method=method
                    )

                    # Important: Always start from a fresh copy of the original trained model
                    model_to_quantize = copy.deepcopy(original_model_instance).to(device)
                    
                    try:
                        quantized_model = quantize_simple_cnn_model(
                            model_to_quantize,
                            loftq_args,
                            quantize_final_fc_layer=final_quant_status
                        )
                        
                        accuracy = evaluate_model(quantized_model, testloader, device=device)
                        
                        # Conceptual: Prepare for LoRA fine-tuning if needed
                        # quantized_model = prepare_for_lora_fine_tuning(quantized_model, final_layer_name='fc2_final', final_layer_quantized_with_lora=final_quant_status)
                        # Then you would train LoRA adapters and re-evaluate.
                        # For this script, we are evaluating the "post-LoftQ-initialization" accuracy.

                    except Exception as e:
                        print(f"ERROR during quantization/evaluation for {bits}-bit, rank {rank}, {method}, final_quant={final_quant_status}: {e}")
                        accuracy = "Error" # Log error in CSV

                    end_time = time.time()
                    time_taken = end_time - start_time

                    # Append results to CSV
                    with open(csv_file_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([bits, rank, method, final_quant_status, accuracy, f"{time_taken:.2f}"])
                    
                    print(f"Accuracy: {accuracy} (Time: {time_taken:.2f}s)")

    print("\nParameter sweep finished. Results saved to:", csv_file_path)
