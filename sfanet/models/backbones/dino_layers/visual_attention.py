# !/bin/bash

import torch
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torch import Tensor
import torch.nn.functional as F


class VisualAttention(nn.Module):
    def __init__(self):
        super(VisualAttention, self).__init__()

        self.queries = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.keys = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.values = nn.Conv2d(3, 1, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        q = self.queries(x)
        k = self.keys(x)
        v = self.values(x)

        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = torch.softmax(attn_weights, dim=-1)

        attended_values = torch.matmul(attn_weights, v).contiguous()
        return attended_values
