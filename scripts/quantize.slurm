#!/bin/bash

#SBATCH --nodes=1
#SBATCH -p lyceum
#SBATCH --gres=gpu:3

#SBATCH --output=../../output.txt
#SBATCH --error=../../output.txt
#SBATCH --time=15:00:00


cd ..
python quantize_save.py \
    --model_name_or_path NousResearch/Llama-2-7b-hf \
    --bits 2 \
    --iter 5 \
    --rank 16 \
    --save_dir "$HF_HOME/loftq/"


## quantization_list=(2 2.25 2.5 3 4 16)
#quantization_list=(2 4 8)
#
#for bit in "${quantization_list[@]}"; do
#  echo "Saving $bit-bit-rank16 quantized model"
#  python quantize_save.py \
#    --model_name_or_path NousResearch/Llama-2-7b-hf \
#    --bits "$bit" \
#    --iter 5 \
#    --rank 16 \
#    --save_dir "$HF_HOME/loftq/"
#done

