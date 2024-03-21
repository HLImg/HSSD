# -*- coding: utf-8 -*-
# @Time : 2024/03/07 19:40
# @Author : Liang Hao
# @FileName : arch_unet.py
# @Email : lianghao@whu.edu.cn

import torch
import torch.nn as nn

from .arch_module import Mambaformer
from .arch_utils import OverlapPatchEmbed, Upsample3D, Downsample3D

class MambaFormerUnet(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 out_channels=1,
                 dim = 16, 
                 num_blocks = [4, 6, 6, 8],
                 num_refine_block = 4,
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
        
        self.patch_embed = OverlapPatchEmbed(in_channels, embed_dim=dim)
        
        self.encoder_level1 = nn.Sequential(
            *[Mambaformer(dim=dim, s_expand=s_expand, 
                          d_state=d_state, 
                          d_expand=d_expand, 
                          d_conv=d_conv,
                          drop_prob=drop_prob, act_name=act_name, 
                          conv_name=conv_name, ffn_expand=ffn_expand, 
                          ffn_name=ffn_name, bias=bias) for i in range(num_blocks[0])]
        )
        self.down1_2 = Downsample3D(dim)
        
        self.encoder_level2 = nn.Sequential(
            *[Mambaformer(dim=int(dim * (2 ** 1)), s_expand=s_expand, 
                          d_state=d_state, 
                          d_expand=d_expand, 
                          d_conv=d_conv,
                          drop_prob=drop_prob, act_name=act_name, 
                          conv_name=conv_name, ffn_expand=ffn_expand, 
                          ffn_name=ffn_name, bias=bias) for i in range(num_blocks[1])]
        )
        self.down2_3 = Downsample3D(int(dim * (2 ** 1)))
        
        self.encoder_level3 = nn.Sequential(
            *[Mambaformer(dim=int(dim * (2 ** 2)), s_expand=s_expand, 
                          d_state=d_state, 
                          d_expand=d_expand, 
                          d_conv=d_conv,
                          drop_prob=drop_prob, act_name=act_name, 
                          conv_name=conv_name, ffn_expand=ffn_expand, 
                          ffn_name=ffn_name, bias=bias) for i in range(num_blocks[2])]
        )
        self.down3_4 = Downsample3D(int(dim * (2 ** 2)))
        
        self.latent = nn.Sequential(
            *[Mambaformer(dim=int(dim * (2 ** 3)), s_expand=s_expand, 
                          d_state=d_state, 
                          d_expand=d_expand, 
                          d_conv=d_conv,
                          drop_prob=drop_prob, act_name=act_name, 
                          conv_name=conv_name, ffn_expand=ffn_expand, 
                          ffn_name=ffn_name, bias=bias) for i in range(num_blocks[3])]
        )
        
        self.up4_3 = Upsample3D(int(dim * (2 ** 3)))
        self.reduce_chan_level3 = nn.Conv3d(int(dim * (2 ** 3)), int(dim * (2 ** 2)), 
                                            kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(
            *[Mambaformer(dim=int(dim * (2 ** 2)), s_expand=s_expand, 
                          d_state=d_state, 
                          d_expand=d_expand, 
                          d_conv=d_conv,
                          drop_prob=drop_prob, act_name=act_name, 
                          conv_name=conv_name, ffn_expand=ffn_expand, 
                          ffn_name=ffn_name, bias=bias) for i in range(num_blocks[2])]
        )
        
        self.up3_2 = Upsample3D(int(dim * (2 ** 2)))
        self.reduce_chan_level2 = nn.Conv3d(int(dim * (2 ** 2)), int(dim * (2 ** 1)), 
                                            kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[Mambaformer(dim=int(dim * (2 ** 1)), s_expand=s_expand, 
                          d_state=d_state, 
                          d_expand=d_expand, 
                          d_conv=d_conv,
                          drop_prob=drop_prob, act_name=act_name, 
                          conv_name=conv_name, ffn_expand=ffn_expand, 
                          ffn_name=ffn_name, bias=bias) for i in range(num_blocks[1])]
        )
        
        self.up2_1 = Upsample3D(int(dim * (2 ** 1)))
        self.decoder_level1 = nn.Sequential(
            *[Mambaformer(dim=int(dim * (2 ** 1)), s_expand=s_expand, 
                          d_state=d_state, 
                          d_expand=d_expand, 
                          d_conv=d_conv,
                          drop_prob=drop_prob, act_name=act_name, 
                          conv_name=conv_name, ffn_expand=ffn_expand, 
                          ffn_name=ffn_name, bias=bias) for i in range(num_blocks[0])]
        )
        
        self.refinement = nn.Sequential(
            *[Mambaformer(dim=int(dim * (2 ** 1)), s_expand=s_expand, 
                          d_state=d_state, 
                          d_expand=d_expand, 
                          d_conv=d_conv,
                          drop_prob=drop_prob, act_name=act_name, 
                          conv_name=conv_name, ffn_expand=ffn_expand, 
                          ffn_name=ffn_name, bias=bias) for i in range(num_refine_block)]
        )
        
        
        self.output = nn.Conv3d(int(dim * (2 ** 1)), out_channels, 3, 1, 1, bias=bias)
    
    @torch.no_grad()
    def forward(self, img):
        img = torch.unsqueeze(img, 1)
        inp_enc_level1 = self.patch_embed(img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 
        
        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)
        
        out_dec_level1 = self.output(out_dec_level1) + img
        
        return out_dec_level1.squeeze(1)