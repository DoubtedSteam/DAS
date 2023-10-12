export CUDA_VISIBLE_DEVICES='3' 
python run_train.py with data_root=./arrows num_gpus=1 num_nodes=1 task_finetune_irtr_f30k_clip_bert get_recall_metric=True \
per_gpu_batchsize=32 clip16 text_roberta image_size=384 test_only=True \
load_path="" skip_module=[]
