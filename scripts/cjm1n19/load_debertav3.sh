# LoftQ: train 4-bit 64-rank llama-2-7b with LoftQ on GSM8K using 8 A100s
# global batch_size=64
python train_model.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --decompose \
  --loftq