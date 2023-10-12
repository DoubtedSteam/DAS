export CUDA_VISIBLE_DEVICES='2' 
python run_train.py with data_root=./arrows num_gpus=1 num_nodes=1  task_finetune_nlvr2_clip_bert per_gpu_batchsize=64 clip16 text_roberta image_size=288 test_only=True \
load_path="" skip_module=[]

