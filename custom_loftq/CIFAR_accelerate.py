import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
from torchvision import datasets, transforms, models
# from torch.utils.data import DataLoader # Trainer handles DataLoader creation
import copy
import csv
import os
import time
from pathlib import Path

from accelerate import Accelerator
from transformers import TrainingArguments, Trainer, EvalPrediction
import evaluate # HF's new library for metrics
import numpy as np

from model_utils import convert_linear_layer, estimate_model_size
from loftq_cnn import TrueQuantizedConv2d, compute_device # compute_device might be superseded by Accelerator
from model_utils import convert_true_quant_conv_layer, estimate_model_size
from loftq import TrueQuantizedLinear


# Initialize accelerator
accelerator = Accelerator()

# --- 1. Define the ResNet50 Model (adapted for CIFAR-10) ---
def get_resnet50_for_cifar10(num_classes=10):
    # Load a ResNet50 model.
    model = models.resnet50(weights=None, num_classes=num_classes) # weights=None for training from scratch

    # Adapt ResNet50 for CIFAR-10:
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
def get_cifar10_datasets(data_dir='./data'):
    Path(data_dir).mkdir(parents=True, exist_ok=True) # Ensure data directory exists
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

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    return train_dataset, test_dataset


# --- Custom Trainer for models that don't output HuggingFace ModelOutput ---
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure criterion is passed or set if not part of the model
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs): # <--- MODIFIED HERE
        # The 'inputs' from our cifar_collate_fn are structured by the collator
        # The default collator or your custom one should provide a dictionary
        images = inputs.get("pixel_values")
        labels = inputs.get("labels")

        if images is None or labels is None:
            raise ValueError(
                "Inputs dictionary must contain 'pixel_values' and 'labels' keys. "
                "Check your data collator and dataset format."
            )

        outputs = model(images)
        loss = self.criterion(outputs, labels)
        return (loss, outputs) if return_outputs else loss

# --- Metrics Computation for Trainer ---
accuracy_metric = evaluate.load("accuracy")

def compute_metrics_fn(p: EvalPrediction): # Renamed to avoid conflict
    preds = np.argmax(p.predictions, axis=1)
    return accuracy_metric.compute(predictions=preds, references=p.label_ids)


# --- 3. Quantization Function ---
def quantize_resnet_model(model_to_quantize, model_args, quantize_final_fc_layer=True):
    accelerator.print(f"\nQuantizing ResNet model... Final FC layer quantization: {quantize_final_fc_layer}")

    final_fc_layer_name = 'fc'
    linear_target_modules = []
    if quantize_final_fc_layer:
        if hasattr(model_to_quantize, final_fc_layer_name) and isinstance(getattr(model_to_quantize, final_fc_layer_name), nn.Linear):
            linear_target_modules.append(final_fc_layer_name)
        else:
            accelerator.print(f"Warning: Final linear layer '{final_fc_layer_name}' not found or not nn.Linear. Not targeting for quantization.")

    linear_excluded_modules = []
    if not quantize_final_fc_layer:
        if hasattr(model_to_quantize, final_fc_layer_name) and isinstance(getattr(model_to_quantize, final_fc_layer_name), nn.Linear):
            linear_excluded_modules.append(final_fc_layer_name)

    # Quantize Convolutional Layers
    model_to_quantize = convert_true_quant_conv_layer(model_to_quantize, model_args)

    # Quantize Linear Layers (if any targeted)
    if linear_target_modules or not quantize_final_fc_layer: # Proceed if targeting or if needing to exclude
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
        accelerator.print("No linear layers targeted or excluded for quantization based on current settings.")
    accelerator.print("Quantization complete.")
    return model_to_quantize

# --- Custom Collator ---
def cifar_collate_fn(batch):
    # Batch is a list of tuples (image, label)
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    # Trainer expects a dictionary
    return {"pixel_values": images, "labels": labels}


# --- 6. Main Experiment Flow ---
if __name__ == '__main__':
    train_dataset, test_dataset = get_cifar10_datasets()
    accelerator.print("\n--- Loaded dataset ---")

    csv_file_path = 'loftq_resnet50_cifar10_hf_results.csv'
    csv_headers = ['int_bit', 'reduced_rank', 'quant_method', 'final_layer_quantized',
                   'accuracy_eval', 'train_time_s', 'eval_time_s', 'quant_time_s', 'train_loss']

    if accelerator.is_main_process:
        Path(csv_file_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_file_path):
            with open(csv_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(csv_headers)

    bits_to_test = [2, 4, 8]
    ranks_to_test = [16, 32] # Reduced for quicker testing
    quant_methods_to_test = ["uniform", "normal"] # Reduced for quicker testing
    final_layer_quant_options = [True, False] # Reduced for quicker testing
    num_train_epochs_sweep = 1 # Reduced for quicker testing
    train_batch_size_sweep = 128
    eval_batch_size_sweep = 256
    learning_rate_sweep = 0.1 # SGD learning rate

    accelerator.print(f"\n\n--- Starting Quantization Parameter Sweep (ResNet50 on CIFAR-10) with Hugging Face Trainer & Accelerate ---")
    accelerator.print(f"Running on device: {accelerator.device}")
    accelerator.print(f"Number of processes: {accelerator.num_processes}")

    trainer, model, optimizer = None, None, None
    for bits in bits_to_test:
        for rank in ranks_to_test:
            for method in quant_methods_to_test:
                for final_quant_status in final_layer_quant_options:
                    run_name = f"b{bits}_r{rank}_m{method}_fcq{str(final_quant_status).lower()}"
                    output_dir = f"./results_loftq_resnet50/{run_name}"
                    logging_dir = f"./logs_loftq_resnet50/{run_name}"

                    # Clean up output and logging directories for fresh run by main process
                    if accelerator.is_main_process:
                        if os.path.exists(output_dir):
                            import shutil
                            shutil.rmtree(output_dir)
                        if os.path.exists(logging_dir):
                            import shutil
                            shutil.rmtree(logging_dir)
                        Path(output_dir).mkdir(parents=True, exist_ok=True)
                        Path(logging_dir).mkdir(parents=True, exist_ok=True)
                    accelerator.wait_for_everyone() # Ensure dirs are made before non-main processes might need them

                    accelerator.print("-" * 60)
                    accelerator.print(f"Testing: Bits={bits}, Rank={rank}, Method='{method}', FinalFCQuant={final_quant_status}, Run: {run_name}")

                    quant_start_time = time.time()
                    model = get_resnet50_for_cifar10() # Instantiate on CPU

                    loftq_args = MockModelArgs(
                        int_bit=bits,
                        reduced_rank=rank,
                        quant_method=method
                    )
                    try:
                        # Quantization happens on CPU before model is sent to accelerator.prepare()
                        model = quantize_resnet_model(
                            model,
                            loftq_args,
                            quantize_final_fc_layer=final_quant_status
                        )
                        quant_end_time = time.time()
                        quant_time_taken = quant_end_time - quant_start_time
                        accelerator.print(f"Quantization time: {quant_time_taken:.2f}s")

                        if accelerator.is_main_process:
                            try:
                                # temp_fp_model = get_resnet50_for_cifar10()
                                # estimate_model_size(temp_fp_model, model)
                                # del temp_fp_model
                                accelerator.print("Model size estimation (if applicable by your function):")
                                estimate_model_size(get_resnet50_for_cifar10(), model)
                            except Exception as est_e:
                                accelerator.print(f"Could not estimate model size: {est_e}")


                        training_args = TrainingArguments(
                            output_dir=output_dir,
                            logging_dir=logging_dir,
                            num_train_epochs=num_train_epochs_sweep,
                            per_device_train_batch_size=max(1, train_batch_size_sweep // accelerator.num_processes),
                            per_device_eval_batch_size=max(1, eval_batch_size_sweep // accelerator.num_processes),
                            gradient_accumulation_steps=1, # Adjust if needed
                            # optim="sgd", # Not needed if passing optimizer instance
                            # learning_rate=learning_rate_sweep, # Set in optimizer instance
                            warmup_steps=0,
                            weight_decay=5e-4, # Used by AdamW if no optimizer specified, or by SGD if param is present
                            logging_strategy="steps",
                            logging_steps=max(1, int(len(train_dataset) / (train_batch_size_sweep * accelerator.num_processes * 0.1 ))), # Log ~10 times per epoch
                            eval_strategy="epoch",
                            save_strategy="epoch",
                            save_total_limit=1,
                            load_best_model_at_end=True,
                            metric_for_best_model="accuracy",
                            greater_is_better=True,
                            report_to="tensorboard", # or "wandb" or "none"
                            dataloader_num_workers=2,
                            # ddp_find_unused_parameters=False, # Set if issues with DDP and custom layers
                            label_names=["labels"], # Explicitly tell Trainer what the label column is called
                            remove_unused_columns=False, # Important if dataset returns more than model inputs
                        )

                        optimizer = optim.SGD(model.parameters(), lr=learning_rate_sweep, momentum=0.9, weight_decay=5e-4, nesterov=True)
                        # lr_scheduler = None # Let Trainer handle or define one explicitly

                        trainer = CustomTrainer(
                            model=model, # Trainer will use accelerator.prepare
                            args=training_args,
                            train_dataset=train_dataset,
                            eval_dataset=test_dataset,
                            compute_metrics=compute_metrics_fn,
                            data_collator=cifar_collate_fn,
                            optimizers=(optimizer, None) # (optimizer, lr_scheduler)
                        )

                        train_start_time = time.time()
                        train_result = trainer.train()
                        train_end_time = time.time()
                        train_time_taken = train_end_time - train_start_time
                        avg_train_loss = train_result.training_loss if hasattr(train_result, 'training_loss') else "N/A"
                        accelerator.print(f"Training completed. Time: {train_time_taken:.2f}s, Avg Loss: {avg_train_loss}")

                        eval_start_time = time.time()
                        eval_metrics = trainer.evaluate()
                        eval_end_time = time.time()
                        eval_time_taken = eval_end_time - eval_start_time

                        accuracy = eval_metrics.get("eval_accuracy", 0.0) * 100 # Convert to percentage

                    except Exception as e:
                        accelerator.print(f"ERROR during training/evaluation for {run_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        accuracy = "Error"
                        train_time_taken = "Error"
                        eval_time_taken = "Error"
                        quant_time_taken = quant_time_taken if 'quant_time_taken' in locals() else "Error"
                        avg_train_loss = "Error"


                    if accelerator.is_main_process:
                        with open(csv_file_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                bits, rank, method, final_quant_status,
                                f"{accuracy:.2f}" if isinstance(accuracy, float) else accuracy,
                                f"{train_time_taken:.2f}" if isinstance(train_time_taken, float) else train_time_taken,
                                f"{eval_time_taken:.2f}" if isinstance(eval_time_taken, float) else eval_time_taken,
                                f"{quant_time_taken:.2f}" if isinstance(quant_time_taken, float) else quant_time_taken,
                                f"{avg_train_loss:.4f}" if isinstance(avg_train_loss, float) else avg_train_loss
                            ])

                    accelerator.print(f"Accuracy: {accuracy} (Quant T: {quant_time_taken if isinstance(quant_time_taken, str) else f'{quant_time_taken:.2f}s'}, Train T: {train_time_taken if isinstance(train_time_taken, str) else f'{train_time_taken:.2f}s'}, Eval T: {eval_time_taken if isinstance(eval_time_taken, str) else f'{eval_time_taken:.2f}s'})")

                    del model
                    del trainer
                    del optimizer
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    accelerator.wait_for_everyone() # Ensure cleanup and sync before next iteration

    accelerator.print("\nParameter sweep finished. Results saved to:", csv_file_path)