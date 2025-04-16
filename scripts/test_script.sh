#!/bin/bash

#SBATCH --nodes=1
#SBATCH -p lyceum
#SBATCH --gres=gpu

#SBATCH --output=../../output.txt
#SBATCH --error=../../output.txt
#SBATCH --time=15:00:00

cd ..
python test_gsm8k.py \
    --model_name_or_path LoftQ/Llama-2-7b-hf-4bit-64rank \
    --adapter_name_or_path /scratch/yw16u21/huggingface/adapters/gsm8k \
    --batch_size 16

