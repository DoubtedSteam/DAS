
import torch
from torch import nn
import das
from typing import Optional, Tuple
from  torch.cuda.amp import autocast
import das.pl_eval_model
import loralib as lora


    # replaced_list = []
    # for name, module in llama.named_modules():
        
    #         replaced_list.append(name)
    # for name in replaced_list:
    #     module = getattr(llama, name)
    #     layer = lora.Linear(module.in_features, module.out_features, r=8, lora_alpha=16)
    #     setattr(llama, name, layer)

def set_Lora(model, dim=8, lora_list=['wq', 'wk', 'wv', 'wo', 'q_proj', 'k_proj', 'v_proj', 'c_proj']):
    for name, module in model.named_children():
        if any(key in name for key in lora_list):
            layer = lora.Linear(module.in_features, module.out_features, r=dim, lora_alpha=16)
            setattr(model, name, layer)
        elif len(list(module.children())) != 0:
            set_Lora(module, dim)

