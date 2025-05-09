#!/bin/bash -l
#SBATCH -p lyceum
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=be1g21@soton.ac.uk
#SBATCH --time=20:00:00 # Adjust time if 25 epochs take longer

module load conda/py3-latest
conda activate loftq

accelerate launch \
    run_loftq.py \
    --model_name_or_path microsoft/deberta-v3-base \
    --data_name glue \
    --task_name mrpc \
    --num_train_epochs 60 \
    --learning_rate 1e-4 \
    --loftq \
    --decompose \
    --reduced_rank 16\
    --num_iter 5 \
    --int_bit 4 \
    --quant_method normal \

