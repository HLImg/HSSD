# -*- coding: utf-8 -*-
# @Time : 2024/03/07 18:47
# @Author : Liang Hao
# @FileName : arch_module.py
# @Email : lianghao@whu.edu.cn

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from timm.models.layers import DropPath
from einops.layers.torch import Rearrange

from .arch_ffn import FeedForward3D
from .arch_attn import LocalMambaMixing3D


class Mambaformer(nn.Module):
    def __init__(self, 
                 dim, 
                 s_expand = 2, 
                 d_state = 16, 
                 d_conv = 4, 
                 d_expand = 2, 
                 drop_prob = 0., 
                 act_name = 'relu',
                 conv_name = 's3conv',
                 ffn_expand = 2, 
                 ffn_name = 'mlp_gate',
                 bias = False
                 ):
        super().__init__()
        
        self.norm1 = nn.Sequential(
            Rearrange('b c d h w -> b d h w c'),
            nn.LayerNorm(dim),
            Rearrange('b d h w c -> b c d h w')
        )
        
        self.mixer = LocalMambaMixing3D(
            dim=dim, s_expand=s_expand, d_state=d_state, d_conv=d_conv, 
            d_expand=d_expand, bias=bias, act_name=act_name, conv_name=conv_name
        )
        
        self.norm2 = nn.Sequential(
            Rearrange('b c d h w -> b d h w c'),
            nn.LayerNorm(dim),
            Rearrange('b d h w c -> b c d h w')
        )
        
        self.ffn = FeedForward3D(
            dim=dim, expand_factor=ffn_expand, ffn_name=ffn_name
        )
        
        self.drop_path = DropPath(drop_prob=drop_prob)
        
    def forward(self, x):
        # x = self.norm1(x)
        # x = self.mixer(x)
        # print(x.shape)
        x = x + self.drop_path(self.mixer(self.norm1(x)))
        # x = x + self.drop_path(cp.checkpoint(self._checkpointed_forward, x))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x
    
    def _checkpointed_forward(self, x):
        x = self.mixer(self.norm1(x))
        return x