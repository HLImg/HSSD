# -*- coding: utf-8 -*-
# @Time : 2024/03/07 22:23
# @Author : Liang Hao
# @FileName : arch_rir.py
# @Email : lianghao@whu.edu.cn

import torch
import torch.nn as nn

from .arch_module import Mambaformer
from .arch_utils import OverlapPatchEmbed


class ResidualGroup(nn.Module):
    def __init__(self,
                 num_blk, 
                 dim, 
                 drop_prob = 0., 
                 act_name = 'relu',
                 conv_name = 's3conv',
                 ffn_expand = 2, 
                 ffn_name = 'mlp_gate',
                 s_expand = 2, 
                 d_state = 16, 
                 d_conv = 4, 
                 d_expand = 2,
                 bias = False
                 ):
        super().__init__()
        
        
        module_body = [
            Mambaformer(dim=dim, s_expand=s_expand, 
                          d_state=d_state, 
                          d_expand=d_expand, 
                          d_conv=d_conv,
                          drop_prob=drop_prob, act_name=act_name, 
                          conv_name=conv_name, ffn_expand=ffn_expand, 
                          ffn_name=ffn_name, bias=bias) for i in range(num_blk)
        ]
        
        module_body.append(
            nn.Conv3d(dim, dim, 3, 1, 1, bias=True)
        )
        
        self.body = nn.Sequential(*module_body)
        
    def forward(self, x):
        return self.body(x) + x

class MambaFormerRIR(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 out_channels=1,
                 dim = 16, 
                 num_groups = [4, 6, 6, 8],
                 bias=False,
                 act_name = 'relu',
                 conv_name = 's3conv',
                 ffn_expand = 2, 
                 ffn_name = 'mlp_gate',
                 drop_prob = 0., 
                 s_expand = 2, 
                 d_state = 16, 
                 d_conv = 4, 
                 d_expand = 2
                 ):
        
        super().__init__()
        
        module_head = [nn.Conv3d(in_channels, dim, 3, 1, 1, bias=True)]
        module_body = []
        
        drops = [x.item() for x in torch.linspace(0, drop_prob, sum(num_groups))]
        
        for i in range(len(num_groups)):
            module_body.append(
                ResidualGroup(num_blk=num_groups[i], dim=dim, drop_prob=drops[i], 
                              act_name=act_name, conv_name=conv_name, ffn_expand=ffn_expand,
                              ffn_name=ffn_name, s_expand=s_expand, d_state=d_state,
                              d_conv=d_conv, d_expand=d_expand, bias=bias)
            )
            
        self.conv_out = nn.Conv3d(dim, dim, 3, 1, 1, bias=True)
        
        module_tail = [nn.Conv3d(dim, out_channels, 3, 1, 1, bias=True)]
        
        self.head = nn.Sequential(*module_head)
        self.body = nn.Sequential(*module_body)
        self.tail = nn.Sequential(*module_tail)
    
    def forward(self, x):
        head = self.head(x)
        res = self.body(head)
        res = self.tail(self.conv_out(head + res))  + x
        return res