import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import get_activation
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform


def init_bert_weights(module):
    """Initialize the weights."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # std defaults to 0.02, this might need to be changed
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class ParallelAdapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=768,
                 bottleneck=96,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck
        # self.non_linearity = args.non_linearity  # use ReLU by default

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            self.apply(init_bert_weights)
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class Adapter(nn.Module):
    def __init__(self, dim=768):
        super().__init__()

        self.input_dim = dim
        # reduction_factor = 8
        # self.down_sample_size = self.input_dim // reduction_factor
        self.down_sample_size = 256
        self.activation = nn.ReLU(inplace=True)
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size) 
        nn.init.normal_(self.down_sampler.weight, std=1e-2)
        # nn.init.normal_(self.down_sampler.weight, std=1.0)
        nn.init.zeros_(self.down_sampler.bias)
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim)
        nn.init.normal_(self.up_sampler.weight, std=1e-2)
        # nn.init.zeros_(self.up_sampler.weight)
        nn.init.zeros_(self.up_sampler.bias)

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        z = self.up_sampler(z)
        output = x + z
        return output
    

class Gumbel_Adapter(nn.Module):
    def __init__(self, dim=768):
        super().__init__()

        self.input_dim = dim
        # reduction_factor = 8
        # self.down_sample_size = self.input_dim // reduction_factor
        self.down_sample_size = 256
        self.activation = nn.ReLU(inplace=True)
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size) 
        nn.init.normal_(self.down_sampler.weight, std=1e-2)
        nn.init.zeros_(self.down_sampler.bias)
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim)
        nn.init.normal_(self.up_sampler.weight, std=1e-2)
        nn.init.zeros_(self.up_sampler.bias)

        self.weight = nn.Parameter(torch.ones(2) * 0.5)

    def gumbel_softmax(self, logits, dim=-1, temperature=0.1):
        '''
        Use this to replace argmax
        My input is probability distribution, multiply by 10 to get a value like logits' outputs.
        '''
        gumbels = -torch.empty_like(logits).exponential_().log()
        logits = (logits.log_softmax(dim=dim) + gumbels) / temperature
        return F.softmax(logits, dim=dim)

    def get_weights(self, tau):
        logits = torch.softmax(self.weight, dim=-1)
        return self.gumbel_softmax(logits, temperature=tau)
    
    def get_gate(self):
        return self.weight[0] < self.weight[1]

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        z = self.up_sampler(z)
        output = x + z
        return output


# 选择最显著的channel
def Significant_init(original, transform, left=True):
    norm = original.pow(2)
    if left:
        norm = norm.sum(1)
    else:
        norm = norm.sum(0)

    _, index = torch.sort(norm)
    index = index[:]
    index, _ = torch.sort(index)

    transform = torch.zeros_like(transform)
    if left:
        for i in range(transform.shape[1]):
            transform[index[i], i] = 1
    else:
        for i in range(transform.shape[0]):
            transform[i, index[i]] = 1

    return transform


# 紧凑模块，将self-attention的输入进行调整，降低模型的宽度
class CompactAttentionBlock(nn.Module):
    def __init__(
        self,
        input_dim=768,
        hid_dim=576,
    ):
        super().__init__()

        self.hid_dim = hid_dim
        
        self.query = nn.Linear(input_dim, hid_dim)
        self.key = nn.Linear(input_dim, hid_dim)
        self.value = nn.Linear(input_dim, hid_dim)

        nn.init.normal_(self.query.weight, mean=0.0, std=1.0)
        self.query.weight.data = torch.softmax(self.query.weight.data / math.sqrt(input_dim), dim=-1)
        self.key.weight.data = self.query.weight.data
        self.value.weight.data = self.query.weight.data

        nn.init.constant_(self.query.bias, 0.0)
        nn.init.constant_(self.key.bias, 0.0)
        nn.init.constant_(self.value.bias, 0.0)
        
        
# 紧凑模块，将Linear层紧凑化
class CompactLinearBlock(nn.Module):
    def __init__(
        self,
        input_dim=768,
        output_dim=576,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()

        self.dense = nn.Linear(input_dim, output_dim)

        nn.init.normal_(self.dense.weight, mean=0.0, std=1.0)
        self.dense.weight.data = torch.softmax(self.dense.weight.data, dim=0)

        nn.init.constant_(self.dense.bias, 0.0)


class METERcontroller(nn.Module):
    def __init__(
        self
    ):
        super().__init__()

        self.compact_dim = 576
        self.compact_head = 12
        self.compact = (self.compact_dim, self.compact_head)

        self.self_attention = nn.ModuleDict({
            'Attention': CompactAttentionBlock(768, self.compact_dim),
            'Output': CompactLinearBlock(self.compact_dim, 768)
        })

        self.cross_attention = nn.ModuleDict({
            'Attention': CompactAttentionBlock(768, self.compact_dim),
            'Output': CompactLinearBlock(self.compact_dim, 768)
        })

        # self.feedforward = nn.ModuleDict({
        #     'Intermediate': CompactLinearBlock(768*4, self.compact_dim*4),
        #     'Output': CompactLinearBlock(self.compact_dim*4, 768*4)
        # })

        self.feedforward = nn.ModuleDict({
            'Intermediate': None,
            'Output': None
        })


# 调用模块
class LightWeightBlock(nn.Module):
    def __init__(
        self,
        pretrained_module,
        transformation_module,
    ):
        super().__init__()

        self.P = pretrained_module
        self.T = transformation_module

        self.Bias_up = nn.Parameter(torch.FloatTensor(1, 1, self.T.hid_dim * 4))
        self.Bias_down = nn.Parameter(torch.FloatTensor(1, 1, 768))

        nn.init.constant_(self.Bias_up, 0)
        nn.init.constant_(self.Bias_down, 0)

        self.block_grad = []
        self.block_forward = []

    def attn(self, x, mask=None):
        B, N, C = x.shape

        # Self-Attention
        # qkv = torch.matmul(
        #     self.P.attn.qkv.weight.t(), # [C, 3C]
        #     self.T.Tqkv # [3C, 3H]
        # )
        qkv = self.P.attn.qkv.weight.t()
        qkv = torch.cat(
            [
                torch.matmul(qkv[:, :self.P.dim], self.T.Tq),
                torch.matmul(qkv[:, self.P.dim: self.P.dim*2], self.T.Tk),
                torch.matmul(qkv[:, self.P.dim*2: self.P.dim*3], self.T.Tv)
            ],
            dim=1
        )
        qkv = (
            torch.matmul(x, qkv)
            .reshape(B, N, 3, self.P.attn.num_heads, self.T.hid_dim // self.P.attn.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.P.attn.scale
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.P.attn.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.T.hid_dim)

        o = torch.matmul(
            self.T.To, # [H, C]
            self.P.attn.proj.weight.t() # [C, C]
        )
        x = torch.matmul(x, o)

        x = self.P.attn.proj_drop(x)

        return x, attn
    
    def mlp(self, x):
        W_up = torch.matmul(
            self.P.mlp.fc1.weight.t(), # [C, 4C]
            self.T.Tup # [4C, 4H]
        )

        x = torch.matmul(x, W_up) + self.Bias_up
        x = self.P.mlp.act(x)
        x = self.P.mlp.drop(x)

        W_down = torch.matmul(
            self.T.Tdown, # [4H, 4C]
            self.P.mlp.fc2.weight.t() # [4C, C]
        )
        
        x = torch.matmul(x, W_down) + self.Bias_down

        x = self.P.mlp.drop(x)

        return x
    
    def forward(self, x, mask=None):
        _x, attn = self.attn(self.P.norm1(x), mask=mask)
        # print(x)
        x = x + self.P.drop_path(_x)
        x = x + self.P.drop_path(self.mlp(self.P.norm2(x)))
        # print(x)
        # exit()
        return x, attn