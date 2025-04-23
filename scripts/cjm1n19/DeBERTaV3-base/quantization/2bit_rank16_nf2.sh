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

SAVE_DIR="~/LoftQ/models/"
python quantize_save.py \
    --model_name_or_path microsoft/deberta-v3-base \
    --token HF_TOKEN \
    --bits 2 \
    --iter 5 \
    --rank 16 \
    --save_dir $SAVE_DIR
