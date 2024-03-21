# -*- coding: utf-8 -*-
# @Time : 2024/03/07 16:30
# @Author : Liang Hao
# @FileName : arch_attn.py
# @Email : lianghao@whu.edu.cn

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from mamba_ssm import Mamba
from .arch_utils import act
from einops.layers.torch import Rearrange

from .conv3d import build_conv3d


class LocalMambaMixing3D(nn.Module):
    """
    the shape of input features must be (b, c, d, h, w)
    """
    def __init__(self, dim, s_expand=2, d_state=16, d_conv=4, d_expand=2, 
                 bias=False, act_name='relu', conv_name='s3conv'):
        super().__init__()
        
        hidden_dim = int(dim * s_expand)
        
        self.proj_in = nn.Conv3d(dim, hidden_dim * 2, kernel_size=1, bias=bias)
        self.conv_3x3 = nn.Conv3d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, 
                                  stride=1, padding=1, bias=bias, groups=hidden_dim * 2)
        
        # temporal modeling use 1D Mamba
        
        # 可以加入混合机制
        self.temporal = nn.Sequential(
            # (b, c, d, h, w) -> (b, c, d, 1, 1)
            nn.Conv3d(hidden_dim, hidden_dim, 1, bias=False), 
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # (b, c, d, 1, 1) -> (b d, c), 这对应于mamba中的(B, L, D), L表示序列长度，D是通道数量
            Rearrange('b c d h w -> b (d h w) c'),
            # 时序建模：(b, d, c) -> (b, d, c)
            Mamba(d_model=hidden_dim, 
                  d_conv=d_conv, 
                  d_state=d_state, 
                  expand=d_expand,
                  use_fast_path=False,
                #   device=torch.device('cpu')
                  ),
            Rearrange('b (d h w) c -> b c d h w', h=1, w=1),
            nn.Conv3d(hidden_dim, hidden_dim, 1, bias=False), 
            nn.Sigmoid()
        )
                
        self.spatial_local_1 = build_conv3d(conv_name=conv_name)(
            hidden_dim, hidden_dim, kernel=3, stride=1, padding=1, bias=bias, act_name=act_name
        )
        
        self.proj_out = nn.Conv3d(hidden_dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
    
    def forward(self, inp):
        """
        inp: (b, c, d, h, w)
        """
        # x : (b, 4c, d, h, w)
        x = self.proj_in(inp)
        # x1, x2: (b, 2c, d, h, w)
        x1, x2 = self.conv_3x3(x).chunk(2, dim=1)
        # spatial modeling
        x1 = self.spatial_local_1(x1)
        # temporal modeling
        # x2: (b, 2c, d, h, w) -> (b, 2c, d, 1, 1)
       
        x2 = self.temporal(x2)
        # x2 = cp.checkpoint(self._checkpoint, x2)
        
        out = self.proj_out(x1 * x2)
        return out
    
    def _checkpoint(self, x):
        return self.temporal(x)