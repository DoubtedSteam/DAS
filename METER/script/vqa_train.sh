export MASTER_ADDR='0.0.0.0'
export MASTER_PORT='8003'
export NODE_RANK='0'
export CUDA_VISIBLE_DEVICES='3'
python run_train.py with data_root=./arrows num_gpus=1 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=32 \
load_path='meter_clip16_288_roberta_pretrain.ckpt' clip16 text_roberta image_size=384 clip_randaug \
skip_module=[]
