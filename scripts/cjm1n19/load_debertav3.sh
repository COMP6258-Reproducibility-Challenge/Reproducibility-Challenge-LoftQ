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

# LoftQ: train 4-bit 64-rank llama-2-7b with LoftQ on GSM8K using 8 A100s
# global batch_size=64
python train_model.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --decompose \
  --loftq
