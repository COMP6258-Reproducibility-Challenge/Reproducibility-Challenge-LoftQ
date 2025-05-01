python quantize_save.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --token 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 \
    --bits 4 \
    --iter 1 \
    --rank 64 \
    --save_dir ./quantized_models/llama_7b/ 