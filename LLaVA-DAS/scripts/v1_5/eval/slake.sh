export CUDA_VISIBLE_DEVICES=0

python -m llava.eval.model_vqa_kzx \
    --model-path ./checkpoints/llava-v1.5-slake-skip4 \
    --question-file /data/qiong_code/data/slake/Slake1.0/testval_llava_en_new.json \
    --image-folder /data/qiong_code/data/slake/Slake1.0/imgs \
    --answers-file ./checkpoints/llava-v1.5-slake_0.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_kzx.py \
    --gt /data/qiong_code/data/slake/Slake1.0/testval_llava_en.json \
    --pred ./checkpoints/llava-v1.5-slake_0.jsonl
