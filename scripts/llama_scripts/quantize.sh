#!/bin/bash -l
#SBATCH -J llama7b_quant       
#SBATCH -p lyceum              
#SBATCH --mem=64G              
#SBATCH --gres=gpu:4           
#SBATCH --nodes=1              
#SBATCH -c 16               
#SBATCH --mail-type=ALL        
#SBATCH --mail-user=be1g21@soton.ac.uk 
#SBATCH --time=48:00:00        

# --- Environment Setup ---
module load conda/py3-latest
conda activate loftq

# --- Set Paths ---
PROJECT_DIR="${HOME}/LoftQ_Private"
DOWNLOADED_MODEL_PATH="/mainfs/scratch/be1g21/models/llama-2-7b-hf"
QUANTIZED_SAVE_DIR="/mainfs/scratch/be1g21/quantized_models/llama_7b_4bit_64rank/"

# --- Create Output Directory ---
mkdir -p ${QUANTIZED_SAVE_DIR}

# --- Navigate to Project Directory ---
cd ${PROJECT_DIR}

# --- Run Quantization ---
python ./quantize_save.py \
    --model_name_or_path ${DOWNLOADED_MODEL_PATH} \
    --bits 4 \
    --iter 1 \
    --rank 64 \
    --save_dir ${QUANTIZED_SAVE_DIR}

# --- Job End ---
EXIT_CODE=$?
exit ${EXIT_CODE}
