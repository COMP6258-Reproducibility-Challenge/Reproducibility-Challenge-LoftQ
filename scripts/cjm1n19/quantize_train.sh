python quantize_train.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --data_name=wikitext-2 \
    --task_name=mnli \
    --decompose \
    --loftq \
    --reduced_rank 64 \
    --num_iter 1 \
    --int_bit 4 \
    --from_saved