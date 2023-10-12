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


class HyperNetwork(nn.Module):
    def __init__(
        self,
        original_input_dim,
        original_output_dim,
        input_dim = None,
        output_dim = None,
    ):
        super().__init__()

        self.input_dim = input_dim if input_dim is not None else original_input_dim
        self.output_dim = output_dim if output_dim is not None else original_output_dim

        if input_dim is not None:
            self.InputHyperGeneration = nn.Sequential(
                nn.Linear(original_input_dim, input_dim),
                # nn.ReLU(inplace=True),
                # nn.Linear(input_dim * 2, input_dim)
            )
        else:
            self.InputHyperGeneration = None

        if output_dim is not None:
            self.OutputGeneration = nn.Sequential(
                nn.Linear(original_output_dim, output_dim),
                # nn.ReLU(inplace=True),
                # nn.Linear(output_dim * 2, output_dim)
            )
        else:
            self.OutputGeneration = None

    def forward(self, weight, bias):
        if self.InputHyperGeneration is not None:
            weight = self.InputHyperGeneration(weight)

        if self.OutputGeneration is not None:
            weight = weight.t()
            weight = self.OutputGeneration(weight)
            weight = weight.t()
            bias = self.OutputGeneration(bias)

        return weight, bias


class HyperLinear(nn.Module):
    def __init__(
        self,
        original_linear,
        generator
    ):
        super().__init__()

        self.register_buffer('original_weight', original_linear.weight.data.clone())
        self.register_buffer('original_bias', original_linear.bias.data.clone())

        self.Lora = nn.Sequential(
            nn.Linear(generator.input_dim, 4, bias=False),
            nn.Linear(4, generator.output_dim, bias=False)
        )
        nn.init.normal_(self.Lora[0].weight)
        nn.init.zeros_(self.Lora[1].weight)

        self.generator = generator

        self.weight = None
        self.bias = None

    def update(self):
        self.weight, self.bias = self.generator(self.original_weight, self.original_bias)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias) + self.Lora(x)