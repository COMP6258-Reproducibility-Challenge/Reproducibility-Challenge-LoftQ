import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models # Import models for ResNet50
from torch.utils.data import DataLoader
import copy
import csv
import os
import time

# Assuming your LoftQ modules are in the current directory or accessible in PYTHONPATH
from loftq_cnn import TrueQuantizedConv2d, compute_device
from model_utils import convert_true_quant_conv_layer
from loftq import TrueQuantizedLinear
from model_utils import convert_linear_layer

# --- 1. Define the ResNet50 Model (adapted for CIFAR-10) ---
def get_resnet50_for_cifar10(num_classes=10):
    # Load a ResNet50 model. We won't use pre-trained weights from ImageNet directly
    # as we are changing the architecture slightly and training from scratch on CIFAR-10.
    # If you want to use pre-trained weights, you'd need a more complex transfer learning setup.
    model = models.resnet50(weights=None, num_classes=num_classes)

    # Adapt ResNet50 for CIFAR-10:
    # 1. The first conv layer: ImageNet models use a 7x7 kernel with stride 2.
    #    For CIFAR-10 (32x32), this is too aggressive. Change to 3x3 kernel, stride 1.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 2. Remove the initial MaxPool: The original ResNet50 has a MaxPool after the first conv.
    #    For CIFAR-10, this might reduce dimensions too quickly. We can make it an Identity.
    model.maxpool = nn.Identity()
    
    # 3. The final fully connected layer (model.fc) is already set to num_classes
    #    by the num_classes argument in models.resnet50.
    #    If you loaded a pre-trained model with 1000 classes, you would reinitialize it:
    #    num_ftrs = model.fc.in_features
    #    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# --- Mock Model Arguments for LoftQ ---
class MockModelArgs:
    def __init__(self, int_bit=4, reduced_rank=16, num_iter=1, quant_method="normal", true_quantization=True):
        self.int_bit = int_bit
        self.reduced_rank = reduced_rank
        self.num_iter = num_iter
        self.quant_method = quant_method
        self.true_quantization = true_quantization

# --- 2. Data Loading (CIFAR-10) ---
def get_cifar10_loaders(train_batch_size=128, test_batch_size=100):
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandAugment(num_ops=2, magnitude=9), # RandAugment can be quite strong
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_dataset = datasets.CIFAR10(root='data/', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='data/', train=False, download=True, transform=test_transform)

    trainloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2, pin_memory=True) # Reduced num_workers
    testloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2, pin_memory=True) # Reduced num_workers
    return trainloader, testloader


def train_model(model, trainloader, criterion, optimizer, epochs=20, device=compute_device, model_path='resnet50_cifar10_original.pth', scheduler=None):
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded pre-trained original model weights from {model_path}.")
            return model
        except Exception as e:
            print(f"Could not load model from {model_path}: {e}. Training from scratch.")

    model.train()
    print(f"Starting training on {device} for {epochs} epochs...")
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (i + 1) % 100 == 0:
                print(f'[Epoch {epoch + 1}/{epochs}, Batch {i + 1:5d}] loss: {running_loss / 100:.3f} | Acc: {100.*correct/total:.2f}% ({correct}/{total})')
                running_loss = 0.0
        
        epoch_accuracy = 100. * correct / total
        print(f"Epoch {epoch + 1} finished. Training Accuracy: {epoch_accuracy:.2f}%")

        if scheduler:
            scheduler.step()

    print('Finished Training')
    torch.save(model.state_dict(), model_path)
    print(f"Saved trained model weights to {model_path}.")
    return model

# --- 3. Quantization Function ---
def quantize_resnet_model(model_to_quantize, model_args, quantize_final_fc_layer=True):
    print(f"\nQuantizing ResNet model... Final FC layer quantization: {quantize_final_fc_layer}")
    
    # For Linear layers - in ResNet50 from torchvision, the final layer is named 'fc'
    final_fc_layer_name = 'fc' # Default name in torchvision ResNets
    
    linear_target_modules = []
    if quantize_final_fc_layer:
        # Check if the model actually has a layer named final_fc_layer_name
        if hasattr(model_to_quantize, final_fc_layer_name) and isinstance(getattr(model_to_quantize, final_fc_layer_name), nn.Linear):
            linear_target_modules.append(final_fc_layer_name)
        else:
            print(f"Warning: Final linear layer '{final_fc_layer_name}' not found or not nn.Linear. Not targeting for quantization.")
            
    linear_excluded_modules = []
    if not quantize_final_fc_layer:
        linear_excluded_modules.append(final_fc_layer_name)

    # Quantize Convolutional Layers
    # convert_true_quant_conv_layer will find all nn.Conv2d, including those in ResNet blocks
    model_to_quantize = convert_true_quant_conv_layer(model_to_quantize, model_args)

    # Quantize Linear Layers (if any targeted)
    if linear_target_modules:
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
    else:
        print("No linear layers targeted for quantization based on current settings.")
    print("Quantization complete.")
    return model_to_quantize

# --- 5. Evaluation ---
@torch.no_grad()
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
    
    if total == 0:
        return 0.0
    accuracy = 100 * correct / total
    return accuracy

# --- 6. Main Experiment Flow ---
if __name__ == '__main__':
    device = compute_device
    trainloader, testloader = get_cifar10_loaders(train_batch_size=256, test_batch_size=100) # Smaller batch for ResNet if memory is an issue
    
    original_model_path = 'resnet50_cifar10_original.pth' # Changed model path
    original_model_instance = get_resnet50_for_cifar10().to(device) # Use ResNet50
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer and Scheduler for ResNet
    optimizer = optim.SGD(original_model_instance.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True) # Common for ResNet
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer) # Train for more epochs

    # Train or load the original model
    original_model_instance = train_model(original_model_instance, trainloader, criterion, optimizer, epochs=300, device=device, model_path=original_model_path, scheduler=scheduler) # Increased epochs
    
    print("\n--- Evaluating Original Model ---")
    original_accuracy = evaluate_model(original_model_instance, testloader, device=device)
    print(f"Original Model Accuracy (ResNet50 on CIFAR-10): {original_accuracy:.2f} %")

    # --- CSV Setup ---
    csv_file_path = 'loftq_resnet50_cifar10_results.csv' # Changed CSV name
    csv_headers = ['int_bit', 'reduced_rank', 'quant_method', 'final_layer_quantized', 'accuracy', 'time_taken_quant_eval_s']
    
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

    # --- Parameter Ranges for Sweep ---
    bits_to_test = [4, 8] # Starting with 4 and 8 bits for ResNet50; 2-bit might be very challenging
    ranks_to_test = [16, 32] # ResNet50 layers are larger, might benefit from slightly larger ranks
    quant_methods_to_test = ["uniform", "normal"]
    final_layer_quant_options = [True, False]

    print("\n\n--- Starting Quantization Parameter Sweep (ResNet50 on CIFAR-10) ---")

    for bits in bits_to_test:
        for rank in ranks_to_test:
            for method in quant_methods_to_test:
                for final_quant_status in final_layer_quant_options:
                    start_time = time.time()
                    print("-" * 60)
                    print(f"Testing: Bits={bits}, Rank={rank}, Method='{method}', FinalFCQuant={final_quant_status}")

                    loftq_args = MockModelArgs(
                        int_bit=bits,
                        reduced_rank=rank, 
                        quant_method=method
                    )

                    model_to_quantize = get_resnet50_for_cifar10().to(device) # Use ResNet50
                    
                    try:
                        quantized_model = quantize_resnet_model( # Use the new quantization function
                            model_to_quantize,
                            loftq_args,
                            quantize_final_fc_layer=final_quant_status
                        )

                        quantized_model = train_model(quantized_model, trainloader, criterion, optimizer, epochs=300, device=device, model_path=original_model_path, scheduler=scheduler) # Increased epochs
                        
                        accuracy = evaluate_model(quantized_model, testloader, device=device)
                        
                    except Exception as e:
                        print(f"ERROR during quantization/evaluation for {bits}-bit, rank {rank}, {method}, final_quant={final_quant_status}: {e}")
                        import traceback
                        traceback.print_exc()
                        accuracy = "Error"

                    end_time = time.time()
                    time_taken = end_time - start_time

                    with open(csv_file_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([bits, rank, method, final_quant_status, accuracy, f"{time_taken:.2f}"])
                    
                    print(f"Accuracy: {accuracy} (Time: {time_taken:.2f}s)")

    print("\nParameter sweep finished. Results saved to:", csv_file_path)
