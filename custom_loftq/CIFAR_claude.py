import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import copy
import csv
import os
import time

# Import accelerate library
from accelerate import Accelerator
# Import tqdm for progress bars
from tqdm import tqdm

# LoftQ imports
from loftq_cnn import TrueQuantizedConv2d, compute_device
from model_utils import convert_true_quant_conv_layer
from loftq import TrueQuantizedLinear
from model_utils import convert_linear_layer, estimate_model_size

# --- 1. Define the ResNet50 Model (adapted for CIFAR-10) ---
def get_resnet50_for_cifar10(num_classes=10):
    # Load a ResNet50 model without pre-trained weights
    model = models.resnet50(weights=None, num_classes=num_classes)

    # Adapt ResNet50 for CIFAR-10
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    
    return model

# --- Mock Model Arguments for LoftQ ---
class MockModelArgs:
    def __init__(self, int_bit=4, reduced_rank=16, num_iter=5, quant_method="normal", true_quantization=True):
        self.int_bit = int_bit
        self.reduced_rank = reduced_rank
        self.num_iter = num_iter
        self.quant_method = quant_method
        self.true_quantization = true_quantization

# --- 2. Data Loading (CIFAR-10) ---
def get_cifar10_loaders(accelerator, train_batch_size=128, test_batch_size=100):
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Only download on main process
    download = accelerator.is_main_process
    
    # Make sure the directory exists
    os.makedirs('data/', exist_ok=True)
    
    # Download dataset (only on main process)
    if download:
        accelerator.print("Downloading CIFAR-10 dataset...")
    
    train_dataset = datasets.CIFAR10(root='data/', train=True, download=download, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='data/', train=False, download=download, transform=test_transform)
    
    # Make sure all processes can access the dataset after main process has downloaded it
    accelerator.wait_for_everyone()
    
    trainloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader


def train_model(accelerator, model, trainloader, criterion, optimizer, epochs=20, model_path='resnet50_cifar10_original.pth', scheduler=None):
    # Get the device from accelerator
    device = accelerator.device
    
    model.train()
    accelerator.print(f"Starting training with Accelerate on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Use tqdm for progress tracking if main process
        if accelerator.is_main_process:
            pbar = tqdm(total=len(trainloader), desc=f"Epoch {epoch+1}/{epochs}")
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            
            # No need for .to(device) - accelerator handles this
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Use accelerator for backward pass
            accelerator.backward(loss)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar on main process
            if accelerator.is_main_process:
                pbar.update(1)
                if (i + 1) % 100 == 0:
                    pbar.set_postfix({"loss": running_loss / 100, "acc": 100.*correct/total})
                    running_loss = 0.0
        
        # Close progress bar
        if accelerator.is_main_process:
            pbar.close()
            
        epoch_accuracy = 100. * correct / total
        accelerator.print(f"Epoch {epoch + 1} finished. Training Accuracy: {epoch_accuracy:.2f}%")

        # Pass metrics to scheduler if it's ReduceLROnPlateau
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_accuracy)
            else:
                scheduler.step()

    accelerator.print('Finished Training')
    
    # Save the model - use accelerator's save method to handle distributed setups
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    
    if accelerator.is_main_process:
        torch.save(unwrapped_model.state_dict(), model_path)
        accelerator.print(f"Saved trained model weights to {model_path}.")
    
    return model

# --- 3. Quantization Function ---
def quantize_resnet_model(model_to_quantize, model_args, quantize_final_fc_layer=True):
    print(f"\nQuantizing ResNet model... Final FC layer quantization: {quantize_final_fc_layer}")
    
    final_fc_layer_name = 'fc'
    
    linear_target_modules = []
    if quantize_final_fc_layer:
        if hasattr(model_to_quantize, final_fc_layer_name) and isinstance(getattr(model_to_quantize, final_fc_layer_name), nn.Linear):
            linear_target_modules.append(final_fc_layer_name)
        else:
            print(f"Warning: Final linear layer '{final_fc_layer_name}' not found or not nn.Linear. Not targeting for quantization.")
            
    linear_excluded_modules = []
    if not quantize_final_fc_layer:
        linear_excluded_modules.append(final_fc_layer_name)

    # Quantize Convolutional Layers
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
def evaluate_model(accelerator, model, testloader):
    model.eval()
    correct = 0
    total = 0
    
    # Use tqdm for progress tracking if main process
    if accelerator.is_main_process:
        pbar = tqdm(total=len(testloader), desc=f"Evaluating")
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # No need for .to(device) as accelerator handles this
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar on main process
            if accelerator.is_main_process:
                pbar.update(1)
    
    # Close progress bar
    if accelerator.is_main_process:
        pbar.close()
    
    # Gather results across all processes
    # Create tensors for gathering
    correct_tensor = torch.tensor([correct], device=accelerator.device)
    total_tensor = torch.tensor([total], device=accelerator.device)
    
    # Gather them
    all_correct = accelerator.gather(correct_tensor).sum().item()
    all_total = accelerator.gather(total_tensor).sum().item()
    
    if all_total == 0:
        return 0.0
    accuracy = 100 * all_correct / all_total
    return accuracy

# --- 6. Main Experiment Flow ---
def main():
    # Initialize the accelerator
    accelerator = Accelerator()
    
    # Get dataloaders with accelerator reference
    train_batch_size = 16  # Adjust batch size based on your GPU memory
    test_batch_size = 100
    trainloader, testloader = get_cifar10_loaders(accelerator, train_batch_size=train_batch_size, test_batch_size=test_batch_size)
    
    accelerator.print("\n--- Loaded dataset ---")
    
    original_model_path = 'resnet50_cifar10_original.pth'
    
    # Create model and prepare for distributed training
    original_model_instance = get_resnet50_for_cifar10()
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer and Scheduler for ResNet
    optimizer = optim.SGD(original_model_instance.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    # Prepare model, optimizer, dataloader with accelerator
    trainloader, testloader, original_model_instance, optimizer = accelerator.prepare(
        trainloader, testloader, original_model_instance, optimizer
    )
    
    # CSV setup - only write from main process
    csv_file_path = 'loftq_resnet50_cifar10_results.csv'
    csv_headers = ['int_bit', 'reduced_rank', 'quant_method', 'final_layer_quantized', 'accuracy', 'time_taken_quant_eval_s']
    
    if accelerator.is_main_process and not os.path.exists(csv_file_path):
        with open(csv_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
    
    # Make sure all processes have the dataset
    accelerator.wait_for_everyone()

    # Parameter Ranges for Sweep
    bits_to_test = [4, 8]
    ranks_to_test = [16, 32]
    quant_methods_to_test = ["uniform", "normal"]
    final_layer_quant_options = [True, False]

    accelerator.print("\n\n--- Starting Quantization Parameter Sweep (ResNet50 on CIFAR-10) ---")

    for bits in bits_to_test:
        for rank in ranks_to_test:
            for method in quant_methods_to_test:
                for final_quant_status in final_layer_quant_options:
                    start_time = time.time()
                    accelerator.print("-" * 60)
                    accelerator.print(f"Testing: Bits={bits}, Rank={rank}, Method='{method}', FinalFCQuant={final_quant_status}")

                    loftq_args = MockModelArgs(
                        int_bit=bits,
                        reduced_rank=rank, 
                        quant_method=method
                    )

                    # Create a fresh model for quantization
                    model_to_quantize = get_resnet50_for_cifar10()
                    
                    try:
                        # Quantize model (before prepare)
                        quantized_model = quantize_resnet_model(
                            model_to_quantize,
                            loftq_args,
                            quantize_final_fc_layer=final_quant_status
                        )
                        
                        if accelerator.is_main_process:
                            estimate_model_size(get_resnet50_for_cifar10(), quantized_model)

                        # Create optimizer for quantized model
                        optimizer = optim.SGD(quantized_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
                        
                        # Prepare the model and optimizer for distributed training
                        quantized_model, optimizer = accelerator.prepare(quantized_model, optimizer)
                        
                        # Make sure all processes sync here
                        accelerator.wait_for_everyone()
                        
                        # Train the quantized model
                        quantized_model = train_model(
                            accelerator, 
                            quantized_model, 
                            trainloader, 
                            criterion, 
                            optimizer, 
                            epochs=3, 
                            model_path=f'resnet50_cifar10_quantized_b{bits}_r{rank}_{method}_{final_quant_status}.pth',
                            scheduler=scheduler
                        )
                        
                        # Evaluate the quantized model
                        accuracy = evaluate_model(accelerator, quantized_model, testloader)
                        
                    except Exception as e:
                        accelerator.print(f"ERROR during quantization/evaluation for {bits}-bit, rank {rank}, {method}, final_quant={final_quant_status}: {e}")
                        import traceback
                        traceback.print_exc()
                        accuracy = "Error"

                    end_time = time.time()
                    time_taken = end_time - start_time

                    # Write results to CSV from main process only
                    if accelerator.is_main_process:
                        with open(csv_file_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([bits, rank, method, final_quant_status, accuracy, f"{time_taken:.2f}"])
                    
                    accelerator.print(f"Accuracy: {accuracy} (Time: {time_taken:.2f}s)")

    accelerator.print("\nParameter sweep finished. Results saved to:", csv_file_path)


if __name__ == '__main__':
    main()