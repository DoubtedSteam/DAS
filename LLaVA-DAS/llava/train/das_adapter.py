import torch
import transformers

from torch import nn
from typing import Optional, Tuple
from  torch.cuda.amp import autocast

import llava

import torch.nn.functional as F


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

        self.down = nn.Linear(in_features, hidden_dim)
        self.act = nn.ReLU()
        self.up = nn.Linear(hidden_dim, in_features)

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        with autocast():
            out = self.down(x)
            out = self.act(out)
            out = self.up(out)
        
        return x + out


def forward_llama(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        question_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        if self.skipped_flag < 0:
            residual = hidden_states
            
            hidden_states = self.input_layernorm(hidden_states)
            adapter_states = self.adapter_adapt_0(hidden_states).type_as(residual)
            
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=adapter_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = residual + hidden_states

            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            adapter_states = self.adapter_adapt_1(hidden_states).type_as(residual)
            hidden_states = self.mlp(adapter_states)
            hidden_states = residual + hidden_states
        else:
            residual = hidden_states
            
            hidden_states = self.input_layernorm(hidden_states)
            adapter_states = self.adapter_replace(hidden_states).type_as(residual)
            hidden_states = residual + adapter_states
            
            self_attn_weights = torch.zeros(1, 1, 1, 1)
            present_key_value = torch.zeros(1, 1, 1, 1)
            
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
            
        if self.training:
            outputs += (0, )

        return outputs


def set_Adapter(model, dim_adapt, dim_replace):
    for _ in model.children():
        if type(_) == llava.model.language_model.modeling_llama.LlamaDecoderLayer:
            _.dim = 4096
            _.adapter_adapt_0 = Adapter(_.dim, hidden_dim=dim_adapt // 2)
            _.adapter_adapt_1 = Adapter(_.dim, hidden_dim=dim_adapt // 2)
            _.adapter_replace = Adapter(_.dim, hidden_dim=dim_replace)
            _.skipped_flag = -1.
            
            bound_method = forward_llama.__get__(_, _.__class__)            
            setattr(_, 'forward', bound_method)
            
        elif len(list(_.children())) != 0:
            set_Adapter(_, dim_adapt, dim_replace)
            
