#!/bin/bash -l
#SBATCH -p lyceum
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cjm1n19@soton.ac.uk
#SBATCH --time=12:00:00
#SBATCH -o %j-squad_2_16_uniform_16_32_normal.out

module load conda/py3-latest
conda activate comp6258_env

accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --mixed_precision no \
    run_loftq.py \
    --model_name_or_path microsoft/deberta-v3-base \
    --data_name squad \
    --task_name plain_text \
    --decompose \
    --loftq \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --reduced_rank 16 \
    --quant_method uniform \
    --num_iter 5 \
    --int_bit 2 \
    --gradient_accumulation_steps 4 \
    --save_steps 2500

accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --mixed_precision no \
    run_loftq.py \
    --model_name_or_path microsoft/deberta-v3-base \
    --data_name squad \
    --task_name plain_text \
    --decompose \
    --loftq \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --reduced_rank 16 \
    --quant_method normal \
    --num_iter 5 \
    --int_bit 2 \
    --gradient_accumulation_steps 4 \
    --save_steps 2500

accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --mixed_precision no \
    run_loftq.py \
    --model_name_or_path microsoft/deberta-v3-base \
    --data_name squad \
    --task_name plain_text \
    --decompose \
    --loftq \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --reduced_rank 32 \
    --quant_method normal \
    --num_iter 5 \
    --int_bit 2 \
    --gradient_accumulation_steps 4 \
    --save_steps 2500
