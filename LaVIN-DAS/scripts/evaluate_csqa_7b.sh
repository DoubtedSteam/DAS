CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node 1 --master_port 11113 eval_mmlu.py \
    --ckpt_dir ../data/weights/ \
    --llm_model 7B\
    --tokenizer_path ../data/weights/tokenizer.model \
    --data_root ../data \
    --caption_file ../data/captions.json \
    --adapter_path ./outputs/csqa/6/checkpoint-4.pth \
    --adapter_type attn \
    --adapter_dim 8 \
    --adapter_scale 1 \
    --prompt_format QCM-ALE \
    --max_batch_size 8 \
    --max_seq_len 4096 \
    --split test \
    --n_prompt 6 \
    --temperature 10.\
    --visual_adapter_type router \
    --language_dataset csqa-zs \
    --skip_list '[18, 26, 22, 19, 29, 27]'



# 23, 19
