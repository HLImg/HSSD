# -*- coding: utf-8 -*-
# @Time : 2024/03/07 18:13
# @Author : Liang Hao
# @FileName : arch_utils.py
# @Email : lianghao@whu.edu.cn

import torch
import torch.nn as nn

from einops.layers.torch import Rearrange

def act(name='relu'):
    if name.lower() == 'relu':
        return nn.ReLU()
    elif name.lower() == 'leakyrelu':
        return nn.LeakyReLU()
    elif name.lower() == 'gelu':
        return nn.GELU()
    elif name.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif name.lower() == 'identity':
        return nn.Identity()
    else:
        raise ValueError(f"Activation Function Error, recevied {name}")
    

class Downsample3D(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        
        self.body = nn.Sequential(
            nn.Conv3d(n_feat, n_feat // 2, 3, 1, 1, bias=False),
            Rearrange('b c d h w -> b d c h w'),
            nn.PixelUnshuffle(2),
            Rearrange('b d c h w -> b c d h w')
        )
        
        self.n_feat = n_feat
    
    def forward(self, x):
        return self.body(x)


class Upsample3D(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        
        self.body = nn.Sequential(
            nn.Conv3d(n_feat, n_feat * 2, 3, 1, 1, bias=False),
            Rearrange('b c d h w -> b d c h w'),
            nn.PixelShuffle(2),
            Rearrange('b d c h w -> b c d h w')
        )
    
    def forward(self, x):
        return self.body(x)
    
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch=1, embed_dim=16, bias=False):
        super().__init__()
        
        self.proj = nn.Conv3d(in_ch, embed_dim, 3, 1, 1, bias=bias)
        
    def forward(self, x):
        x = self.proj(x)
        return x
    