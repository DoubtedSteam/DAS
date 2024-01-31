# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from das.pl_eval_model import ModelArgs, Transformer
from das.tokenizer import Tokenizer
from das.pl_generator import LLaMA_Generator
from das.pl_adapter import set_PLAdapter
from das.lora import set_Lora
from util.base_prompt import build_prompt
from dataclasses import dataclass
import re
import random

import warnings
import pandas as pd
from PIL import Image

from torchvision.transforms import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from pathlib import Path
import fairscale.nn.model_parallel.initialize as fs_init
import torch.distributed as dist
from util.apply_delta import apply_model_delta_online
from engine import clean_flag, apply_flag

from datasets import load_dataset, Dataset

warnings.filterwarnings('ignore')


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

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

    mp_world_size = fs_init.get_model_parallel_world_size()
    mp_rank = fs_init.get_model_parallel_rank()
    if mp_world_size == len(checkpoints):
        print('same number of shards of checkpoints and training, loading directly...')
        dist.barrier()
        print('[rank=%d, mp_rank=%d] loading from %s' % (dist.get_rank(), mp_rank, checkpoints[mp_rank]))
        checkpoint = torch.load(checkpoints[mp_rank], map_location='cpu')
    else:
        print('different number of shards of checkpoints and training, redistributing...')
        if dist.get_rank() == 0:
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

            full_state_dict_meta = dict((k, v.shape) for k, v in full_state_dict.items())
            dist.broadcast_object_list([full_state_dict_meta, split_dims], src=0)

        else:  # dist.get_rank() != 0
            recv_objs = [None, None]
            dist.broadcast_object_list(recv_objs, src=0)
            full_state_dict_meta, split_dims = recv_objs

        local_state_dict = {}
        for k in sorted(full_state_dict_meta.keys()):
            print('redistributing weights: %s' % k)
            if dist.get_rank() == 0:
                value = full_state_dict[k].cuda().half()
                del full_state_dict[k]
            else:
                value = torch.empty(full_state_dict_meta[k], device='cuda', dtype=torch.half)
            dist.broadcast(value, src=0)
            value = value.cpu()
            if split_dims[k] < 0:
                local_state_dict[k] = value
            else:
                dim = split_dims[k]
                assert dim >= 0 and dim < value.ndim and value.size(dim) % mp_world_size == 0
                shard_size = value.size(dim) // mp_world_size
                shard_st, shard_ed = shard_size * mp_rank, shard_size * (mp_rank + 1)
                # TODO: make more general
                if dim == 0:
                    value = value[shard_st: shard_ed]
                elif dim == 1:
                    value = value[:, shard_st: shard_ed]
                else:
                    raise NotImplementedError()
                local_state_dict[k] = value.clone()

        checkpoint = local_state_dict

    return checkpoint, tokenizer, params


def print_scores(scores):
    latex_output = ""
    for key, score in scores.items():
        print(f"{key[4:]}: \t{score}")
        latex_output += f"& {score} "
    latex_output += "\\\\"
    print(latex_output)

def load(
    ckpt_dir: str,
    llm_model:str,
    tokenizer_path: str,
    adapter_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
    adapter_type: str,
    adapter_dim:int,
    adapter_scale:float,
    hidden_proj:int,
    visual_adapter_type: str,
    temperature: float,
    use_vicuna: bool,
    skip_list:str='[]',
) -> LLaMA_Generator:
    start_time = time.time()
    checkpoint, tokenizer, params = _load_and_redistribute_checkpoint(ckpt_dir, llm_model)

    print("Loading")
    adapter_checkpoint = torch.load(adapter_path, map_location="cpu")

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size,hidden_proj=hidden_proj, **params
    )
    model_args.vocab_size = tokenizer.n_words

    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    set_Lora(model, 4)
    # set_PLAdapter(model, adapter_type, dim=adapter_dim, s=adapter_scale,t=temperature)

    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    if use_vicuna:
        apply_model_delta_online(model,'../data/weights/vicuna_'+llm_model)

    state_dict = {}
    for key in adapter_checkpoint['model']:
        state_dict[key.replace('module.','')]=adapter_checkpoint['model'][key]

    model.load_state_dict(state_dict, strict=False)
    model.to(torch.device('cuda'))

    clean_flag(model)  
    select = skip_list #[int(i) for i in skip_list[1:-1].split(',')]
    print(select)
    apply_flag(model, select)
    generator = LLaMA_Generator(model, tokenizer)
    
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    adapter_path: str,
    data_root:str,
    caption_file:str,
    max_seq_len: int,
    max_batch_size: int,
    llm_model:str='7B',
    generation_temperature: float = 0.1,
    top_p: float = 0.75,
    split='val',
    prompt_format='QCM-ALE',
    use_caption=False,
    options=["A", "B", "C", "D", "E"],
    adapter_type='repattn',
    adapter_dim=8,
    adapter_scale=1,
    n_prompt=10,
    hidden_proj=128,
    visual_adapter_type='normal',
    temperature=10.,
    use_vicuna=False,
    root_dir_='../data/mme',
    language_dataset='mmlu-zs',
    skip_list:str='[]',
):
    print(max_batch_size,max_seq_len)
    print('use caption: ',use_caption)
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, llm_model, tokenizer_path, adapter_path, local_rank, world_size, max_seq_len, max_batch_size,
        adapter_type,adapter_dim,adapter_scale,hidden_proj,visual_adapter_type,
    temperature, use_vicuna, skip_list=skip_list)
    
    if language_dataset == 'mmlu-zs':
        mmlu_dataset = load_dataset("json", data_files={
            'eval': '../data/mmlu/zero_shot_mmlu_val.json',
            'test': '../data/mmlu/zero_shot_mmlu_test.json',
        })
        mmlu_dataset = mmlu_dataset.remove_columns('subject')
    # MMLU Five-shot (Eval/Test only)
    elif language_dataset == 'mmlu' or language_dataset == 'mmlu-fs':
        mmlu_dataset = load_dataset("json", data_files={
            'eval': '../data/mmlu/five_shot_mmlu_val.json',
            'test': '../data/mmlu/five_shot_mmlu_test.json',
        })
        # mmlu_dataset = mmlu_dataset.remove_columns('subject')
    elif language_dataset == 'csqa-fs':
        mmlu_dataset = load_dataset("json", data_files={
            'test': '../data/CommonSenseQA/commonsense_qa_5_shot_test.json'
        })
    elif language_dataset == 'csqa-zs':
        mmlu_dataset = load_dataset("json", data_files={
            'test': '../data/CommonSenseQA/commonsense_qa_0_shot_test.json'
        })
    elif language_dataset == 'boolq-zs':
        mmlu_dataset = load_dataset("json", data_files={
            'test': '../data/BoolQ/boolq_0_shot_test.json'
        })
    elif language_dataset == 'boolq-fs':
        mmlu_dataset = load_dataset("json", data_files={
            'test': '../data/BoolQ/boolq_5_shot_test.json'
        })
    elif language_dataset == 'gsm8k-zs':
        mmlu_dataset = load_dataset("json", data_files={
            'test': '../data/GSM8k/gsm8k_0_shot_test.json'
        })
    elif language_dataset == 'gsm8k-fs':
        mmlu_dataset = load_dataset("json", data_files={
            'test': '../data/GSM8k/gsm8k_5_shot_test.json'
        })
        

    mmlu_dataset = mmlu_dataset['test']
    # if args.max_mmlu_samples is not None:
    #     mmlu_dataset = mmlu_dataset.select(range(args.max_mmlu_samples))
    # abcd_idx = [
    #     generator.tokenizer.encode("A", bos=False, eos=False)[0],#.input_ids,
    #     generator.tokenizer.encode("B", bos=False, eos=False)[0],#.input_ids[0],
    #     generator.tokenizer.encode("C", bos=False, eos=False)[0],#.input_ids[0],
    #     generator.tokenizer.encode("D", bos=False, eos=False)[0],#.input_ids[0],
    # ]
    # abcd = ['A', 'B', 'C', 'D']

    if 'gsm8k' in language_dataset:
        pattern = re.compile(r'answer is [^\d]*([\d,.]+)\s*[A-Za-z]*\.?')
        addition_patter = None
    else:
        pattern = re.compile(r'answer is ([A-Z]).')
        addition_patter = re.compile(r'Answer: ([A-Z])')
    
    
    import time
    
    start_time = time.time()
    
    total_items = len(mmlu_dataset['input'])
    # total_items = 100
    correct = 0
    for i in range(total_items // max_batch_size + int(total_items % max_batch_size > 0)):
        print("{}/{}".format(i, total_items // max_batch_size + int(total_items % max_batch_size > 0)))
        # subjects = mmlu_dataset['subject'][i * max_batch_size: (i+1) * max_batch_size]
        prompts = mmlu_dataset['input'][i * max_batch_size: (i+1) * max_batch_size]
        answers = mmlu_dataset['output'][i * max_batch_size: (i+1) * max_batch_size]
        
        results, logits = generator.generate(
            prompts, max_gen_len=256 if 'gsm8k' in language_dataset else 10, temperature=generation_temperature, top_p=top_p,
        )

        # print(results)
        # exit()
        
        choices = []
        for k, result in enumerate(results):
            # choices.append(result[1])
            pred = pattern.findall(result)
            if len(pred) == 0 and (addition_patter is not None):
                pred = addition_patter.findall(result)
            
            if len(pred) >= 1:
                pred = pred[0]  # 'A', 'B', ...
                if 'gsm8k' in language_dataset:
                    pred = pred.replace(',', '')
                    if pred[-1]=='.':
                        pred=pred[:-1]
            else:
                print('########')
                print(prompts[k])
                print('########')
                print(result)
                pred = "FAILED"
                print('########')
            choices.append(pred)
            
            # try:
            #     # choices.append(pattern.findall(result)[4])
            # except:
            #     # choices.append('E')
            #     print('########')
            #     print(result)
            #     print('########')
            
        # logits_abcd = [logits[:, ind] for ind in abcd_idx]
        # logits_abcd = torch.stack(logits_abcd, dim=-1)
        # logits_abcd = logits_abcd.argmax(dim=-1)
        # choices = [abcd[ind] for ind in logits_abcd]
        
        print(choices)
        print(answers)
        
        this_part = 0
        for choice, answer in zip(choices, answers):
            this_part += int(choice == answer)
        print(this_part)
        correct += this_part
        print("{:.3f}".format(correct / (max_batch_size * (i+1)) * 100))
        
    print(time.time() - start_time)
        
    print("{:.3f}".format(correct / total_items * 100))

if __name__ == "__main__":
    fire.Fire(main)
