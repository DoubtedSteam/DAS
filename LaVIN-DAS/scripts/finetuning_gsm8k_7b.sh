CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port 11112 train_llama.py \
    --llm_model 7B\
    --llama_model_path ../data/weights/ \
    --data_path ../data/alpaca_data.json \
    --max_seq_len 512 \
    --batch_size 4 \
    --accum_iter 4 \
    --epochs 20 \
    --warmup_epochs 1 \
    --blr 1e-4 \
    --weight_decay 0.02 \
    --output_dir ./outputs/gsm8k/lora \
    --adapter_type attn \
    --adapter_dim 8 \
    --adapter_scale 1 \
    --n_prompt 6 \
    --prompt_format QCM-ALE \
    --temperature 10.\
    --visual_adapter_type router \
    --language_only \
    --language_dataset gsm8k \
    --skip_list '[]' \
    --nas_epoch 0 

# CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node 1 --master_port 11112 eval_mmlu.py \
#     --ckpt_dir ../data/weights/ \
#     --llm_model 7B\
#     --tokenizer_path ../data/weights/tokenizer.model \
#     --data_root ../data \
#     --caption_file ../data/captions.json \
#     --adapter_path ./outputs/gsm8k/lora/checkpoint-19.pth \
#     --adapter_type attn \
#     --adapter_dim 8 \
#     --adapter_scale 1 \
#     --prompt_format QCM-ALE \
#     --max_batch_size 1 \
#     --max_seq_len 4096 \
#     --split test \
#     --n_prompt 6 \
#     --temperature 10.\
#     --visual_adapter_type router \
#     --language_dataset gsm8k-zs \
#     --skip_list '[]'