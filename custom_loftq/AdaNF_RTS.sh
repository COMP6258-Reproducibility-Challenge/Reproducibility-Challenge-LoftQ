#!/bin/bash -l
#SBATCH -p lyceum                 # 🔧 cluster/partition
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --gres=gpu:1              # 1 GPU is enough for DeBERTa‑v3‑base
#SBATCH --mem=48G                 # ↓ (RTE is tiny; 48 GB is plenty)
#SBATCH --time=04:00:00           # ↓ training converges in < 2 h on A100
#SBATCH --mail-type=ALL
#SBATCH --mail-user=be1g21@soton.ac.uk

module load conda/py3-latest
conda activate loftq            # environment that has HF + accelerate + bitsandbytes

# ----------- Hyper‑params -------------------------------------------------
MODEL="microsoft/deberta-v3-base"   # 🔧 same backbone as LoftQ paper
TASK="rte"                          # GLUE RTE
BITS=2                              # 2‑bit quant
PNORM=3                             # AdaNF L3
GRID=12                             # 12 offset candidates / block
RANK=32                             # LoRA rank
ITER=2                              # LoftQ alternations
BATCH=32
ACCUM=1
# -------------------------------------------------------------------------

accelerate launch \
  --num_processes 1 \
  --mixed_precision no \
  run_loftq.py \
  --model_name_or_path "${MODEL}" \
  --data_name glue --task_name "${TASK}" \
  --loftq \
  --quant_method adanf \
  --int_bit "${BITS}" \
  --adanf_pnorm "${PNORM}" \
  --adanf_grid_size "${GRID}" \
  --reduced_rank "${RANK}" \
  --num_iter "${ITER}" \
  --gradient_accumulation_steps "${ACCUM}" \
  --per_device_train_batch_size "${BATCH}" \
  --true_quantization
