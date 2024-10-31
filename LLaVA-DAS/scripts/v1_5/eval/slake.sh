export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_OFFLINE=1

python -m llava.eval.model_vqa_kzx \
    --model-path ./checkpoints/llava-v1.5-slake-skip4 \
    --question-file ./json_data/slake/testval_llava_en_new.json \
    --image-folder /path/to/image \
    --answers-file ./checkpoints/llava-v1.5-slake_1.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_kzx.py \
    --gt ./json_data/slake/testval_llava_en.json \
    --pred ./checkpoints/llava-v1.5-slake_1.jsonl
