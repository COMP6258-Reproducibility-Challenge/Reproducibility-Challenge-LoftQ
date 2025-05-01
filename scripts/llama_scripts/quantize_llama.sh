#!/bin/bash -l
#SBATCH -J llama7b_quant_loftq  # Job name (e.g., llama7b_quant_loftq)
#SBATCH -p gpu                  # Partition: REPLACE <partition_name> with the one containing V100s (e.g., 'gpu', 'v100')
                                # Or fallback to the partition with GTX1080Ti (e.g., 'gtx', 'gpu')
#SBATCH --mem=64G               # System memory request (64GB should be safe for loading)
#SBATCH --gres=gpu:1            # Request 1 GPU (V100 or 1080Ti based on partition)
#SBATCH --nodes=1               # Request 1 node
#SBATCH -c 8                    # Request 8 CPU cores (adjust if needed, usually 4-8 is fine)
#SBATCH --mail-type=ALL         # Send email on job BEGIN, END, FAIL
#SBATCH --mail-user=YOUR_UNI_EMAIL@soton.ac.uk # <<< REPLACE THIS WITH YOUR EMAIL
#SBATCH --time=02:00:00         # Time limit (HH:MM:SS) - 2 hours should be safe, adjust if needed

echo "========================================================"
echo "Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated Memory: $SLURM_MEM_PER_NODE MB"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "========================================================"
echo ""

# --- Environment Setup ---
echo "Loading modules..."
module load conda/py3-latest
echo "Modules loaded."
echo ""

echo "Activating Conda environment..."
# Use the full, correct path to your conda environment
# Replace 'loftq' if your environment name is different
CONDA_ENV_PATH="/lyceum/be1g21/.conda/envs/loftq"
source activate ${CONDA_ENV_PATH}
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment: ${CONDA_ENV_PATH}"
    exit 1
fi
echo "Conda environment activated: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "Transformers version: $(python -c 'import transformers; print(transformers.__version__)')"
echo "PEFT version: $(python -c 'import peft; print(peft.__version__)')"
echo ""

# --- Set Paths ---
# Use absolute paths for clarity in SLURM jobs
# Assumes your LoftQ_Private repo is directly in your home directory
PROJECT_DIR="${HOME}/LoftQ_Private"
# Path where you downloaded the original Llama-2 model using download_model.py
DOWNLOADED_MODEL_PATH="/mainfs/scratch/be1g21/models/llama-2-7b-hf"
# Path where the output (quantized model + adapters) will be saved
QUANTIZED_SAVE_DIR="/mainfs/scratch/be1g21/quantized_models/llama_7b_4bit_64rank/" # Be specific

echo "Project Directory:       ${PROJECT_DIR}"
echo "Downloaded Model Path:   ${DOWNLOADED_MODEL_PATH}"
echo "Quantized Output Dir:    ${QUANTIZED_SAVE_DIR}"
echo ""

# --- Create Output Directory ---
# Important: Create the specific output directory structure needed by quantize_save.py
# The script saves the base model to QUANTIZED_SAVE_DIR and adapters to QUANTIZED_SAVE_DIR/loftq_init
echo "Ensuring base output directory exists: ${QUANTIZED_SAVE_DIR}"
mkdir -p ${QUANTIZED_SAVE_DIR}
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create output directory: ${QUANTIZED_SAVE_DIR}"
    exit 1
fi
echo "Output directory checked/created."
echo ""

# --- Navigate to Project Directory ---
echo "Changing to project directory: ${PROJECT_DIR}"
cd ${PROJECT_DIR}
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to change directory to: ${PROJECT_DIR}"
    exit 1
fi
echo "Current directory: $(pwd)"
echo ""

# --- Run Quantization ---
echo "Starting quantization script (quantize_save.py)..."
# Using the verbose version of quantize_save.py you created
python ./quantize_save_verbose.py \
    --model_name_or_path ${DOWNLOADED_MODEL_PATH} \
    --bits 4 \
    --iter 1 \
    --rank 64 \
    --save_dir ${QUANTIZED_SAVE_DIR} # The script will create subdirs based on model name

# Check exit status
EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Quantization script finished successfully."
else
    echo "ERROR: Quantization script failed with exit code ${EXIT_CODE}."
fi
echo ""

# --- Job End ---
echo "========================================================"
echo "Job Ended: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "========================================================"

exit ${EXIT_CODE}