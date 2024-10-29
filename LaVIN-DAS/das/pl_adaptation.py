
import torch
import json

from das import ModelArgs, Tokenizer, PL_Transformer
from das.pl_adapter import set_PLAdapter
from das.lora import set_Lora

from pathlib import Path
from util.apply_delta import apply_model_delta_online

def _load_and_redistribute_checkpoint(llama_model_path, model_name):
    with open(Path(llama_model_path) / model_name / 'params.json') as f:
        params = json.load(f)
    tokenizer = Tokenizer(model_path=str(Path(llama_model_path) / 'tokenizer.model'))
    print('Using model path: %s, model_name: %s' % (llama_model_path, model_name))
    if model_name=='7B':
        checkpoint = torch.load(llama_model_path + model_name + '/consolidated.00.pth', map_location="cpu")
        return checkpoint, tokenizer, params

    checkpoints = (Path(llama_model_path) / model_name).glob('*.pth')
    checkpoints = sorted(checkpoints)

    loaded = []
    for x in checkpoints:
        print('loading from', x)
        loaded.append(torch.load(x, map_location='cpu'))

    full_state_dict = {}
    split_dims = {}

    def add_weight_with_split_dim(name, dim):
        if dim < 0:  # bcast without split
            full_state_dict[name] = loaded[0][name].clone()
        else:
            full_state_dict[name] = torch.cat([x[name] for x in loaded], dim=dim)
        for x in loaded:
            del x[name]
        split_dims[name] = dim

    add_weight_with_split_dim('tok_embeddings.weight', 1)
    add_weight_with_split_dim('norm.weight', -1)
    add_weight_with_split_dim('output.weight', 0)
    for i in range(params['n_layers']):
        print('gathering layer %d of %d' % (i, params['n_layers']))
        layer_prefix = f'layers.{i}.'
        bcast_names = [
            'attention_norm.weight',
            'ffn_norm.weight',
        ]
        column_parallel_names = [
            'attention.wq.weight',
            'attention.wk.weight',
            'attention.wv.weight',
            'feed_forward.w1.weight',
            'feed_forward.w3.weight',
        ]
        row_parallel_names = [
            'attention.wo.weight',
            'feed_forward.w2.weight',
        ]
        for key in bcast_names:
            add_weight_with_split_dim(layer_prefix + key, -1)
        for key in column_parallel_names:
            add_weight_with_split_dim(layer_prefix + key, 0)
        for key in row_parallel_names:
            add_weight_with_split_dim(layer_prefix + key, 1)

    checkpoint=full_state_dict


    return checkpoint, tokenizer, params

def LLaMA(args):
    llama_model_path = args.llama_model_path
    model_name = args.llm_model

    checkpoint, tokenizer, params = _load_and_redistribute_checkpoint(llama_model_path, model_name)

    model_args: ModelArgs = ModelArgs(
        max_seq_len=args.max_seq_len, max_batch_size=32, hidden_proj=args.hidden_proj, drop_path=args.drop_path, **params
    )

    model_args.vocab_size = tokenizer.n_words

    if args.cpu_load:
        #cpu load is slow, but is freindly for GPU with limited memory.
        torch.set_default_tensor_type(torch.HalfTensor)
    else:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    llama = PL_Transformer(model_args)
    # set_Lora(llama, 4)
  
    torch.set_default_tensor_type(torch.FloatTensor)

    if args.bits in ['4bit','8bit']:
        from util.quantization import quant_model_bnb
        llama.layers=quant_model_bnb(llama.layers,quant_bit=args.bits)

    llama.load_state_dict(checkpoint, strict=False)
    if args.use_vicuna:
        apply_model_delta_online(llama,'../../data/weights/vicuna_'+args.llm_model)

    if args.adapter_type=='block' or  args.adapter_type=='attn':
        set_PLAdapter(llama,args.adapter_type,dim=args.adapter_dim,s=args.adapter_scale,t=args.temperature,gradient_checkpointing=args.gradient_checkpointing)        

    # learnable_keys=['adapter']
    learnable_keys=['lora', 'adapter']
    train_total = 0.
    total = 0.
    trainable_names = []
    for name, param in llama.named_parameters():
        param.requires_grad = False
        
    for name, param in llama.named_parameters():
        total += param.nelement()
        for key in learnable_keys:
            if key in name:
                param.requires_grad = True
                param.data = param.data.float()
                train_total += param.nelement()
                trainable_names.append(name)
    print('  + Number of trainable params: %.2fM' % (train_total / 1e6))
    print('  + Ratio of trainable params: %.2f%%' % (train_total / total * 100))

    if not args.search_mode:
        if args.skip_list != '[]':
            select = [int(i) for i in args.skip_list[1:-1].split(',')]
        else:
            select = []
        for i in range(len(llama.layers)):
            if i in select:
                del llama.layers[i].attention
                del llama.layers[i].feed_forward
                del llama.layers[i].ffn_norm
    total_usage = 0.
    for name, param in llama.named_parameters():
        total_usage += param.nelement()
    print('  + Ratio of deleted params: %.2f%%' % ((1 - total_usage / total) * 100))

    return llama
