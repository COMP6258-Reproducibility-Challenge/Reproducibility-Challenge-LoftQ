#!/bin/bash

# Activate the conda environment
# Make sure conda is initialized in your shell, or source the appropriate profile file first
# Example: source ~/anaconda3/etc/profile.d/conda.sh or source ~/miniconda3/etc/profile.d/conda.sh
conda activate loftq

# --- Configuration Variables ---
# Path to the base model on Hugging Face Hub
BASE_MODEL_PATH="meta-llama/Llama-2-7b-hf"

# Quantization parameters
BITS=4
RANK=64 # Rank used for Llama-2 in the paper's Table 5 [cite: LoftQ_Private/loftq.pdf]
ITER=5  # LoftQ iterations

# Directory to save the quantized model output
SAVE_DIR="./quantized_models/"

# --- Ensure Save Directory Exists ---
mkdir -p $SAVE_DIR

# --- Run the Quantization Script ---
echo "Starting LoftQ quantization for $BASE_MODEL_PATH..."
echo "Config: ${BITS}-bit, Rank-${RANK}, Iterations-${ITER}"
echo "Saving to: $SAVE_DIR"

python quantize_save.py \
    --model_name_or_path $BASE_MODEL_PATH \
    --bits $BITS \
    --rank $RANK \
    --iter $ITER \
    --save_dir $SAVE_DIR

echo "Quantization finished for ${BITS}-bit."
echo "Output saved in: ${SAVE_DIR}${BASE_MODEL_PATH##*/}-${BITS}bit-${RANK}rank/"

