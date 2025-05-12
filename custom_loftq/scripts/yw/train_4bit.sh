#!/bin/bash -l

#SBATCH -p lyceum
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --time=7:00:00
#SBATCH --output=results/train_llama_q4.txt
#SBATCH --error=results/train_llama_q4.txt


module load conda/py3-latest
conda activate comp6258-env

python run_loftq.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --data_name mikasenghaas/wikitext-2 \
    --task_name default \
    --decompose \
    --loftq \
    --reduced_rank 32 \
    --num_iter 1 \
    --int_bit 4 \
    --from_saved
