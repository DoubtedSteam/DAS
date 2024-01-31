
import torch
from torch import nn
import das
from typing import Optional, Tuple
from torch.cuda.amp import autocast
from das.lora import set_Lora 
import das.pl_eval_model


class Adapter(nn.Module):
    def __init__(
        self,
        in_features=768,
        hidden_dim=8,
        groups=2,
        scale=1,
        t=10.
    ):
        super().__init__()


        self.down_sampler = nn.Linear(in_features, hidden_dim, bias=True)
        self.up_sampler = nn.Linear(hidden_dim, in_features, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout(0.1)
        self.scale = scale

        nn.init.xavier_uniform_(self.down_sampler.weight)
        nn.init.zeros_(self.down_sampler.bias)
        nn.init.zeros_(self.up_sampler.weight)
        nn.init.zeros_(self.up_sampler.bias)


    def forward(self, x, res=True, weights=None):
        with autocast():
            if res:
                x = x + self.scale * self.up_sampler(self.relu(self.down_sampler(x)))
            else:
                x = self.scale * self.up_sampler(self.relu(self.down_sampler(x)))
        return x


def forward_llama_block(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
    if self.training and self.gradient_checkpointing:
        if self.skipped_flag < 0:
            h = x + self.drop_path(torch.utils.checkpoint.checkpoint(self.attention, self.adapter_attn(self.attention_norm(x)), start_pos, freqs_cis, mask))
            out = h + self.drop_path(torch.utils.checkpoint.checkpoint(self.feed_forward, self.adapter_mlp(self.ffn_norm(h))))
        else:
            out = x + self.replaced_adapter(self.attention_norm(x), res=False)
    else:
        if self.skipped_flag < 0:
            h = x + self.drop_path(self.attention.forward(self.adapter_attn(self.attention_norm(x)), start_pos, freqs_cis, mask, adapter))
            out = h + self.drop_path(self.feed_forward.forward(self.adapter_mlp(self.ffn_norm(h))))
        else:
            out = x + self.replaced_adapter(self.attention_norm(x), res=False)
    return out


def forward_llama_attn(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
    if self.training and self.gradient_checkpointing:
        if self.skipped_flag < 0:
            h = x + self.drop_path(torch.utils.checkpoint.checkpoint(self.attention, self.adapter_attn(self.attention_norm(x)), start_pos, freqs_cis, mask))
            out = h + self.drop_path(torch.utils.checkpoint.checkpoint(self.feed_forward, self.ffn_norm(h)))
        else:
            out = x + self.replaced_adapter(self.attention_norm(x), res=False)
    else:
        if self.skipped_flag < 0:
            h = x + self.drop_path(self.attention.forward(self.adapter_attn(self.attention_norm(x)), start_pos, freqs_cis, mask, adapter))
            out = h + self.drop_path(self.feed_forward.forward(self.ffn_norm(h)))
        else:
            out = x + self.replaced_adapter(self.attention_norm(x), res=False)
    return out

    
def forward_llama_attn_cache(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
    bs_ = x.shape[0]
    if self.skipped_flag < 0:
        h = x + self.drop_path(self.attention.forward(self.adapter_attn(self.attention_norm(x), weights=self.cache_weights[:bs_]), start_pos, freqs_cis, mask, adapter))
        out = h + self.drop_path(self.feed_forward.forward(self.ffn_norm(h)))
    else:
        out = x + self.replaced_adapter(self.attention_norm(x), res=False, weights=self.cache_weights[:bs_])
    return out


def forward_llama_block_cache(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
    bs_=x.shape[0]
    if self.skipped_flag < 0:
        h = x + self.drop_path(self.attention.forward(self.adapter_attn(self.attention_norm(x), weights=self.cache_weights[:bs_]), start_pos, freqs_cis, mask, adapter))
        out = h + self.drop_path(self.feed_forward.forward(self.adapter_mlp(self.ffn_norm(h), weights=self.cache_weights_ffn[:bs_])))
    else:
        out = x + self.replaced_adapter(self.attention_norm(x), res=False, weights=self.cache_weights[:bs_])
    return out


def set_PLAdapter(model, method, dim=8, s=1, set_forward=True,t=10,gradient_checkpointing=False):
    if method == 'block':
        # not support right now
        assert NotImplementedError
        for _ in model.children():
            if type(_) ==  das.pl_model.TransformerBlock or type(_) == das.pl_eval_model.TransformerBlock:
                _.adapter_attn = Adapter(_.dim, hidden_dim=dim, scale=s, t=t)
                _.adapter_mlp = Adapter(_.dim, hidden_dim=dim, scale=s, t=t)
                _.s = s
                _.t = t
                _.skipped_flag = -1.
                _.replaced_adapter = Adapter(_.dim, hidden_dim=dim * 4, scale=s, t=t)
                _.gradient_checkpointing = gradient_checkpointing
                if type(_) == das.pl_eval_model.TransformerBlock:
                    bound_method = forward_llama_block_cache.__get__(_, _.__class__)
                else:
                    bound_method = forward_llama_block.__get__(_, _.__class__)
                if set_forward:
                    setattr(_, 'forward', bound_method)
            elif len(list(_.children())) != 0:
                set_PLAdapter(_, method, dim, s,set_forward=set_forward,t=t,gradient_checkpointing=gradient_checkpointing)

    else:
        for _ in model.children():
            if type(_) == das.pl_model.TransformerBlock or type(_) == das.pl_eval_model.TransformerBlock:
                _.adapter_attn = Adapter(_.dim,hidden_dim=dim, scale=s, t=t)
                _.s = s
                _.t = t
                _.skipped_flag = -1.
                _.replaced_adapter = Adapter(_.dim, hidden_dim=dim * 4, scale=s, t=t)

                # def get_parameter(model):
                #     return sum(p.numel() for p in model.parameters())

                # print(get_parameter(_.adapter_attn))
                # print(get_parameter(_.replaced_adapter))
                # exit()
                _.gradient_checkpointing = gradient_checkpointing
                if type(_) == das.pl_eval_model.TransformerBlock:
                    bound_method = forward_llama_attn_cache.__get__(_, _.__class__)
                else:
                    bound_method = forward_llama_attn.__get__(_, _.__class__)
                if set_forward:
                    setattr(_, 'forward', bound_method)
            elif len(list(_.children())) != 0:
                set_PLAdapter(_, method, dim, s, set_forward=set_forward,t=t,gradient_checkpointing=gradient_checkpointing)

