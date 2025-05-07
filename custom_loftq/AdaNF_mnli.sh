#!/bin/bash -l
#SBATCH -p lyceum                 # 🔧 cluster/partition
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --gres=gpu:1              # 1 GPU
#SBATCH --mem=64G                 # Increased memory slightly for larger dataset
#SBATCH --time=48:00:00           # !!! Significantly increased time for MNLI !!!
#SBATCH --job-name=AdaNF_MNLI     # Changed job name
#SBATCH --mail-type=ALL
#SBATCH --mail-user=be1g21@soton.ac.uk # Keep your email

module load conda/py3-latest
conda activate loftq            # environment

# ----------- Hyper-params (Updated for MNLI) --------------------------------
MODEL="microsoft/deberta-v3-base"
TASK="mnli"                         # <--- Changed task
BITS=2
PNORM=3
GRID=10                             # <--- Changed grid size based on AdaNF paper
RANK=32
ITER=2
LR="1e-4"                           # <--- Set learning rate based on LoftQ paper
EPOCHS=5                            # <--- Set epochs based on LoftQ paper
BATCH=32                            # Kept batch size (adjust if OOM occurs)
ACCUM=1                             # Kept accumulation (adjust if needed)
# -------------------------------------------------------------------------

echo "Starting AdaNF MNLI run..." # Added echo


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
  --learning_rate "${LR}" \
  --num_train_epochs "${EPOCHS}" \
  --output_dir "/scratch/be1g21/trained_models/adanf_mnli_b${BITS}_r${RANK}_lr${LR}_e${EPOCHS}" \
  --logging_dir "/scratch/be1g21/trainer_outputs/adanf_mnli_logs" \
  --logging_steps 50 \
  --true_quantization