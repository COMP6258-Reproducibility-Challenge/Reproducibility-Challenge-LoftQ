#!/bin/bash -l
#SBATCH -p lyceum
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=be1g21@soton.ac.uk
#SBATCH --time=10:00:00

module load conda/py3-latest
conda activate loftq

accelerate launch \
    --num_processes 1 \
    --mixed_precision no \
    run_gsm8k.py \
    --model_name_or_path meta-llama/llama-2-7b-hf \
    --data_name=gsm8k \
    --decompose \
    --loftq \
    --reduced_rank 64 \
    --num_iter 1 \
    --int_bit 4 \
    --gradient_accumulation_steps 4 \
    --true_quantization \
    --no_train
