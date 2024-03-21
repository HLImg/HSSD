# -*- coding: utf-8 -*-
# @Time : 2024/03/07 15:01
# @Author : Liang Hao
# @FileName : arch_ffn.py
# @Email : lianghao@whu.edu.cn

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange

class FeedForward2D(nn.Module):
    def __init__(self, dim, expand_factor, bias=True, ffn_name="conv2d_gate"):
        """FeedForward Network for 2D data
        the shape for input features must be (b, c, h, w)
        Args:
            dim (int): the number of channels
            ffn_name (bool): the name of ffn. (conv2d_gate | mlp_gate | self_modulated )
        """
        super().__init__()
        
        if ffn_name.lower() == 'conv2d_gate':
            self.ffn = FeedForwardGateConv2d(dim, expand_factor=expand_factor, bias=bias)
        elif ffn_name.lower() == 'mlp_gate':
            self.ffn = nn.Sequential(
                Rearrange('b c h w -> b h w c'),
                FeedForwardGateMLP(dim, expand_factor=expand_factor, bias=bias),
                Rearrange('b h w c -> b c h w')
            )
        elif ffn_name.lower() == 'self_modulated':
            self.ffn = self.ffn = nn.Sequential(
                Rearrange('b c h w -> b h w c'),
                FeedForwardSelfModulated(dim, expand_factor=expand_factor, bias=bias),
                Rearrange('b h w c -> b c h w')
            )
        else:
            raise ValueError(f"only support 'conv2d_gate | mlp_gate | self_modulated', but received {ffn_name}")
        
    
    def forward(self, x):
        return self.ffn(x)


class FeedForward3D(nn.Module):
    def __init__(self, dim, expand_factor, bias=True, ffn_name="conv3d_gate"):
        """FeedForward Network for 3D data
        the shape for input features must be (b, c, d, h, w)
        Args:
            dim (int): the number of channels
            ffn_name (bool): the name of ffn. (conv3d_gate | mlp_gate | self_modulated)
        """
        super().__init__()
        
        if ffn_name.lower() == 'conv3d_gate':
            self.ffn = FeedForwardGateConv3d(dim, expand_factor=expand_factor, bias=bias)
        elif ffn_name.lower() == 'mlp_gate':
            self.ffn = nn.Sequential(
                Rearrange('b c d h w -> b d h w c'),
                FeedForwardGateMLP(dim, expand_factor=expand_factor, bias=bias),
                Rearrange('b d h w c -> b c d h w')
            )
        elif ffn_name.lower() == 'self_modulated':
            self.ffn = self.ffn = nn.Sequential(
                Rearrange('b c d h w -> b d h w c'),
                FeedForwardSelfModulated(dim, expand_factor=expand_factor, bias=bias),
                Rearrange('b d h w c -> b c d h w')
            )
        else:
            raise ValueError(f"only support 'conv_gate | mlp_gate | self_modulated', but received {ffn_name}")
        
    
    def forward(self, x):
        return self.ffn(x)



class FeedForwardGateMLP(nn.Module):
    def __init__(self, dim, expand_factor, bias=True):
        super().__init__()
        
        hidden_dim = int(dim * expand_factor)    
          
        self.proj_in = nn.Linear(dim, hidden_dim * 2, bias=bias)
        
        self.linear_chunk = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=bias)
        
        self.proj_out = nn.Linear(hidden_dim, dim, bias=bias)
        
    def forward(self, x):
        """
        x: (b, ... , c) c denotes the channel dimension.
        """
        x = self.proj_in(x)
        x1, x2 = self.linear_chunk(x).chunk(2, dim=-1)
        # gate mechanism
        x = F.gelu(x1) * x2
        x = self.proj_out(x)
        return x

class FeedForwardGateConv2d(nn.Module):
    """
    Restormer: https://github.com/swz30/Restormer
    """
    def __init__(self, dim, expand_factor=1, bias=True):
        
        super().__init__()
        
        hidden_dim = int(dim * expand_factor)
        
        self.proj_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=bias)
        
        self.conv_chunk = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=1, padding=1,
                                    bias=bias, groups=hidden_dim * 2)
        
        self.proj_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)
    
    def forward(self, x):
        """
        x : (b, c, h, w) c denotes the channel dimension.
        """
        x = self.proj_in(x)
        x1, x2 = self.conv_chunk(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.proj_out(x)
        return x
    

class FeedForwardSelfModulated(nn.Module):
    """
    HSDT: https://arxiv.org/pdf/2303.09040.pdf
    """
    def __init__(self, dim, expand_factor=1, bias=True):
        super().__init__()
        
        
        hidden_dim = int(dim * expand_factor)
        self.w_1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w_3 = nn.Linear(dim, hidden_dim, bias=bias)
    
    def forward(self, input):
        """
        input: (b, ... , c)
        """
        x = self.w_1(input)
        x = F.gelu(x)

        x1 = self.w_2(x)

        x = self.w_3(input)
        x, w = torch.chunk(x, 2, dim=-1)
        x2 = x * torch.sigmoid(w)
  
        return x1 + x2
         

class FeedForwardGateConv3d(nn.Module):
    def __init__(self, dim, expand_factor, bias=True):
        super().__init__()
        
        hidden_dim = int(dim * expand_factor)
        
        self.proj_in = nn.Conv3d(dim, hidden_dim * 2, kernel_size=1, bias=bias)
        
        self.conv_chunck = nn.Conv3d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1, 
                                     stride=1, groups=hidden_dim * 2, bias=bias)
        
        self.proj_out = nn.Conv3d(hidden_dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        """
        x : (b, c, d, h, w)
        """
        x = self.proj_in(x)
        x1, x2 = self.conv_chunck(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.proj_out(x)
        return x

