#!/bin/bash -l

#SBATCH -p swarm_h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=21:00:00
#SBATCH --output=results/train_llama_q16.txt
#SBATCH --error=results/train_llama_q16.txt


# module load conda/py3-latest
conda activate loftq_env

python run_loftq.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --data_name mikasenghaas/wikitext-2 \
    --task_name default \
    --decompose \
    --loftq \
    --reduced_rank 32 \
    --num_iter 1 \
    --int_bit 16 \
    --remove_unused_columns False \
    --true_quantization \
    --from_saved
