#!/bin/bash -l
#SBATCH -p lyceum
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cjm1n19@soton.ac.uk
#SBATCH --time=01:00:00

module load conda/py3-latest
conda activate comp6258_env

export TASK_NAME=mnli
export INT_BIT=4
export LR=1e-4
python run_glue.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --task_name $TASK_NAME \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --num_train_epochs 5 \
  --learning_rate $LR \
  --output_dir output \
  --int_bit $INT_BIT \
  --loftq \
  --reduced_rank 32 \
  --quant_embedding \
  --quant_method uniform \
  --num_iter 5 \
  --decompose
