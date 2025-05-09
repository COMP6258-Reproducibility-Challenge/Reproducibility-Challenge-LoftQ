#!/bin/bash -l
#SBATCH -p lyceum
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=be1g21@soton.ac.uk
#SBATCH --time=10:00:00 # Adjust time if 25 epochs take longer

module load conda/py3-latest
conda activate loftq

accelerate launch \
    run_loftq.py \
    --model_name_or_path facebook/bart-large \
    --data_name xsum \
    --task_name main \
    --num_train_epochs 25 \
    --learning_rate 2e-4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --loftq \
    --decompose \
    --reduced_rank 16 \
    --num_iter 1 \
    --int_bit 4 \
    --quant_method normal \
    --true_quantization \
    --from_saved 

