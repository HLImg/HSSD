# -*- coding: utf-8 -*-
# @Time : 2024/03/07 18:13
# @Author : Liang Hao
# @FileName : conv3d.py
# @Email : lianghao@whu.edu.cn

import torch
import torch.nn as nn

from .arch_utils import act


def build_conv3d(conv_name):
    if conv_name.lower() == 'depthsep':
        return DepthSep3dConv
    elif conv_name.lower() == 's3conv':
        return S3Conv
    else:
        ValueError(f"only support 'depthsep | s3conv ', but received {conv_name}")

class _BaseConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride=1, padding=1, bias=False, act_name='relu'):
        super().__init__()
        
        self.kernel = self._repeat(kernel)
        self.stride = self._repeat(stride)
        self.padding = self._repeat(padding)
        
    def _repeat(self, x):
        if isinstance(x, list):
            return x
        return [x] * 3

class DepthSep3dConv(_BaseConv):
    def __init__(self, in_dim, out_dim, kernel, stride=1, padding=1, bias=False, act_name='relu'):
        super().__init__(in_dim, out_dim, kernel, stride, padding, bias, act_name)
        
        self.dw_conv = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, 
                      kernel_size=(1, self.kernel[1], self.kernel[2]), 
                      stride=(1, self.stride[1], self.stride[2]), 
                      padding=(0, self.padding[1], self.padding[2]), bias=bias),
            act(act_name)
        )
        
        self.pw_conv = nn.Conv3d(out_dim, out_dim, 
                                 kernel_size=(self.kernel[0], 1, 1),
                                 stride=(self.stride[0], 1, 1), 
                                 padding=(self.padding[0], 0, 0),
                                 bias=bias)
    
    def forward(self, x):
        return self.pw_conv(self.dw_conv(x))


class S3Conv(_BaseConv):
    def __init__(self, in_dim, out_dim, kernel, stride=1, padding=1, bias=False, act_name='relu'):
        super().__init__(in_dim, out_dim, kernel, stride, padding, bias, act_name)
        
        self.dw_conv = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, 
                      kernel_size=(1, self.kernel[1], self.kernel[2]), 
                      stride=(1, self.stride[1], self.stride[2]), 
                      padding=(0, self.padding[1], self.padding[2]), bias=bias),
            act(act_name)
        )
        
        self.pw_conv = nn.Conv3d(in_dim, out_dim, 
                                 kernel_size=(self.kernel[0], 1, 1),
                                 stride=(self.stride[0], 1, 1), 
                                 padding=(self.padding[0], 0, 0),
                                 bias=bias)
    
    def forward(self, x):
        x1 = self.dw_conv(x)
        x2 = self.pw_conv(x)
        return x1 + x2
        
        
    
    