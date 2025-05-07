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
    --task_name=sst2 \
    --decompose \
    --loftq \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --reduced_rank 32 \
    --num_iter 5 \
    --int_bit 4 \
    --gradient_accumulation_steps 4 \
    --from_saved \
    --resume_from_checkpoint ./trainer_output/checkpoint-5000 \
    --save_steps 5000
