# !/bin/bash

export CUDA_VISIBLE_DEVICES=4
python -m llava.eval.model_vqa_kzx \
    --model-path ./checkpoints/llava-v1.5-aid20-skip12 \
    --question-file /data/qiong_code/data/aid/test20_qs.json \
    --image-folder /data/qiong_code/data/aid/data/AID \
    --answers-file ./checkpoints/llava-v1.5-aid-20.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1
    
python llava/eval/eval_kzx_closed.py \
    --gt /data/qiong_code/data/aid/test20.json \
    --pred ./checkpoints/llava-v1.5-aid-20.jsonl