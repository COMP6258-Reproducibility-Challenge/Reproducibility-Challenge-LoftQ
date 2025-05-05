accelerate launch ^
    ./run_loftq.py ^
        --model_name_or_path microsoft/deberta-v3-base ^
        --data_name glue ^
        --task_name qqp ^
        --decompose ^
        --num_train_epochs 10 ^
        --learning_rate 5e-5 ^
        --loftq ^
        --reduced_rank 32 ^
        --num_iter 1 ^
        --int_bit 4