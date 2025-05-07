
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
    test_model.py \
    --model_name_or_path microsoft/deberta-v3-base
