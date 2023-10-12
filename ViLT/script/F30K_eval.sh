export CUDA_VISIBLE_DEVICES='0'
python run_train.py with data_root=./arrows num_gpus=1 num_nodes=1 per_gpu_batchsize=4 task_finetune_irtr_f30k_randaug test_only=True precision=32 \
load_path="" skip_module=[]