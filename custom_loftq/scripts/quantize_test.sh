#!/bin/bash -l

#SBATCH -p lyceum
#SBATCH --gres=gpu:4
#SBATCH --nodes=1

module load conda/py3-latest
conda activate comp6258-env

accelerate launch \
  --num_processes 1 \
  --mixed_precision no \
  run_loftq.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --data_name=gsm8k \
  --decompose \
  --loftq \
  --reduced_rank 64 \
  --num_iter 1 \
  --int_bit 8 \
  --gradient_accumulation_steps 4 \
  --true_quantization

