export TASK_NAME=mnli
export INT_BIT=4
export LR=1e-4
python run_glue.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --task_name $TASK_NAME \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 64 \
  --num_train_epochs 5 \
  --learning_rate $LR \
  --output_dir output \
  --int_bit $INT_BIT \
  --loftq \
  --reduced_rank 32 \
  --quant_embedding \
  --quant_method uniform \
  --num_iter 5 \
  --decompose \
  --use_slow_tokenizer