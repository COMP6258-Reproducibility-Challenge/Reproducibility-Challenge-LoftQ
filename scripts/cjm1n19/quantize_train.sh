python quantize_train.py \
    --model_name_or_path microsoft/deberta-v3-base \
    --data_name=glue \
    --task_name=mnli \
    --decompose \
    --loftq \
    --reduced_rank 32 \
    --num_iter 5 \
    --int_bit 4 \
    --from_saved