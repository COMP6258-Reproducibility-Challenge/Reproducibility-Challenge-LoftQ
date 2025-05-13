#!/bin/bash -l
#SBATCH -p lyceum
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cjm1n19@soton.ac.uk
#SBATCH --time=10:00:00

module load conda/py3-latest
conda activate comp6258_env

accelerate launch \
    --num_processes 1 \
    --mixed_precision no \
    run_loftq.py \
    --model_name_or_path microsoft/deberta-v3-base \
    --data_name=glue \
    --task_name=rte \
    --decompose \
    --learning_rate 5e-4 \
    --num_train_epochs 5 \
    --loftq \
    --reduced_rank 32 \
    --num_iter 5 \
    --int_bit 4 \
    --output_dir ./deberta_loftq_rte \
    --gradient_accumulation_steps 4 \

accelerate launch \
    --num_processes 1 \
    --mixed_precision no \
    run_loftq.py \
    --model_name_or_path microsoft/deberta-v3-base \
    --data_name=glue \
    --task_name=rte \
    --decompose \
    --learning_rate 5e-4 \
    --num_train_epochs 5 \
    --loftq \
    --reduced_rank 32 \
    --num_iter 5 \
    --int_bit 4 \
    --output_dir ./deberta_loftq_rte \
    --gradient_accumulation_steps 4 \
    --true_quantization
