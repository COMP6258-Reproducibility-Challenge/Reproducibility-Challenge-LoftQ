#!/bin/bash -l

#SBATCH -p swarm_h100
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=15:00:00
#SBATCH --output=results/train_bart_nf2.txt
#SBATCH --error=results/train_bart_nf2.txt

# module load conda/py3-latest
conda activate loftq_env

# CNN/Daily, rank 8
echo "-------------------- NF2: Training Rank-8 on CNN/Daily --------------------"
python run_loftq.py \
    --model_name_or_path facebook/bart-large \
    --data_name abisee/cnn_dailymail \
    --task_name 1.0.0 \
    --loftq \
    --save_steps 2500 \
    --reduced_rank 8 \
    --num_iter 1 \
    --int_bit 2 \
    --quant_method normal \
    --learning_rate 2e-4 \
    --num_train_epochs 15 \
    --per_device_train_batch_size 64 \
    --output_dir "trained_models"
echo "-------------------- Done --------------------"
echo ""


# CNN/Daily, rank 16
echo "-------------------- NF2: Training Rank-16 on CNN/Daily --------------------"
python run_loftq.py \
    --model_name_or_path facebook/bart-large \
    --data_name abisee/cnn_dailymail \
    --task_name 1.0.0 \
    --loftq \
    --save_steps 2500 \
    --reduced_rank 16 \
    --num_iter 1 \
    --int_bit 2 \
    --quant_method normal \
    --learning_rate 2e-4 \
    --num_train_epochs 15 \
    --per_device_train_batch_size 64 \
    --output_dir "trained_models"
echo "-------------------- Done --------------------"
echo ""


# XSum, rank-8
echo "-------------------- NF2: Training Rank-8 on XSum --------------------"
python run_loftq.py \
    --model_name_or_path facebook/bart-large \
    --data_name xsum \
    --loftq \
    --save_steps 2500 \
    --reduced_rank 8 \
    --num_iter 1 \
    --int_bit 2 \
    --quant_method normal \
    --learning_rate 2e-4 \
    --num_train_epochs 25 \
    --per_device_train_batch_size 32 \
    --output_dir "trained_models"
echo "-------------------- Done --------------------"
echo ""


# XSum, rank-16
echo "-------------------- NF2: Training Rank-16 on XSum --------------------"
python run_loftq.py \
    --model_name_or_path facebook/bart-large \
    --data_name xsum \
    --loftq \
    --save_steps 2500 \
    --reduced_rank 16 \
    --num_iter 1 \
    --int_bit 2 \
    --quant_method normal \
    --learning_rate 2e-4 \
    --num_train_epochs 25 \
    --per_device_train_batch_size 32 \
    --output_dir "trained_models"
echo "-------------------- Done --------------------"
echo ""
