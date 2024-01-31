CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node 2 --master_port 11114 train_llama.py \
    --llm_model 7B\
    --llama_model_path ../data/weights/ \
    --data_path ../data/alpaca_data.json \
    --max_seq_len 512 \
    --batch_size 4 \
    --accum_iter 4 \
    --epochs 5 \
    --warmup_epochs 1 \
    --blr 1e-4 \
    --weight_decay 0.02 \
    --output_dir ./outputs/csqa/0 \
    --adapter_type attn \
    --adapter_dim 8 \
    --adapter_scale 1 \
    --n_prompt 6 \
    --prompt_format QCM-ALE \
    --temperature 10.\
    --visual_adapter_type router \
    --language_only \
    --language_dataset commonsense_qa \
    --skip_list '[]' \
    --nas_epoch 0 


