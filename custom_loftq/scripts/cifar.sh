#!/bin/bash -l
#SBATCH -p lyceum
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=las1g21@soton.ac.uk
#SBATCH --time=10:00:00

module load conda/py3-latest
conda activate PUTYOUR ENV HERE

accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --mixed_precision no \
    CIFAR_no_quant.py
