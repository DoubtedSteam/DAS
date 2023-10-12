export MASTER_ADDR='0.0.0.0'
export MASTER_PORT='9003'
export NODE_RANK='0'
export CUDA_VISIBLE_DEVICES='3' 
python run_train.py with data_root=./arrows num_gpus=1 num_nodes=1 task_finetune_nlvr2_randaug per_gpu_batchsize=64 load_path="./vilt_200k_mlm_itm.ckpt" \
skip_module=[]