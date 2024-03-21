# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/10/30 11:35:33
# @File    :   model.py
# @Contact   :   lianghao@whu.edu.cn

import torch
from net_arch import *
from omegaconf import OmegaConf

class Model:
    def __init__(self, arch_name, device, ckpt):
        ckpt = self.load_ckpt(ckpt)
        self.net = self.__network__(arch_name)
        self.net.load_state_dict(ckpt)
        self.net = self.net.to(device)
        
    
    def __network__(self, name):
        if name.lower() == 'sst_icvl':
            return SST()
        elif name.lower() == 'mamba_icvl':

            return MambaFormerUnet(
               in_channels=1, out_channels=1, dim=16,
               num_blocks=[1, 1, 1, 1], num_refine_block=2, 
               act_name='relu', conv_name='s3conv',
            ffn_expand=2, ffn_name='mlp_gate',
            drop_prob=0.01, s_expand=2, d_conv=4, 
            d_expand=2, d_state=16, bias=False
            )
        
        elif name.lower() == 'sst_real':
            return SST(inp_channels=34,depths=[6,6,6])
        
        elif name.lower() == 'sert_icvl':
            return SERT(inp_channels=31,dim = 96, window_sizes=[16,32,32] , depths=[ 6,6,6], num_heads=[ 6,6,6],
                   split_sizes=[1,2,4],   mlp_ratio=2,   weight_factor=0.1,    memory_blocks=128,   down_rank=8)
            
        elif name.lower() == 'sert_real':
            return SERT(inp_channels=34, dim = 96, window_sizes=[16,32,32] , depths=[6,6,6],
                   down_rank=8, num_heads=[ 6,6,6], split_sizes=[1,2,4], mlp_ratio=2, memory_blocks=64)
            
        elif name.lower() =='hsdt_icvl':
            return HSDT(1, 16, 7, [1, 3, 5])
        
        elif name.lower() == 'grnet_icvl':
            return  U_Net_GR(31, 31)
        
        elif name.lower() == 'grnet_real':
            return  U_Net_GR(34, 34)
        
        elif name.lower() == 'macnet':
            return  MACNet(in_channels=1,channels=16,num_half_layer=5)
        
        elif name.lower() == 't3sc_icvl':
            from omegaconf import OmegaConf
            cfg = OmegaConf.load('./net_arch/t3sc/t3sc_icvl.yaml')
            return MultilayerModel(**cfg.params)
        
        elif name.lower() == 't3sc_real':
            from omegaconf import OmegaConf
            cfg = OmegaConf.load('./net_arch/t3sc/t3sc_real.yaml')
            return MultilayerModel(**cfg.params)
        
        elif name.lower() == 'qrnn3d_icvl':
            return QRNNREDC3D(1, 16, 5, [1,3], has_ad=True)
        
        elif name.lower() == 'proposed_base_real':
            return RIRSCANetv3(34, 32, drop_conv=0., act_conv='relu', num_groups=[6, 6, 6, 6, 6, 6],
                          ca_expand=1, is_group=True, is_bias=False, trans_dim=64, head_dim=32,
                          window_size=8, drop_trans=0.1, input_resolution=128, act_trans='gelu')
        
        elif name.lower() == 'proposed_base_icvl':
            return RIRSCANetv3(31, 32, drop_conv=0., act_conv='relu', num_groups=[6, 6, 6, 6, 6, 6],
                          ca_expand=1, is_group=True, is_bias=False, trans_dim=64, head_dim=32,
                          window_size=8, drop_trans=0.1, input_resolution=64, act_trans='gelu')
        
        elif name.lower() == 'proposed_tiny_icvl':
            return RIRSCANetv3(31, 32, drop_conv=0., act_conv='relu', num_groups=[6, 6, 6],
                          ca_expand=1, is_group=True, is_bias=False, trans_dim=64, head_dim=32,
                          window_size=8, drop_trans=0.1, input_resolution=64, act_trans='gelu')
        
        elif name.lower() == 'local':
            from omegaconf import OmegaConf
            cfg = OmegaConf.load('./net_arch/yaml/local.yaml')
            return RIRLocal(**cfg.net['net_g']['param'])
        
        elif name.lower() == 'global':
            from omegaconf import OmegaConf
            cfg = OmegaConf.load('./net_arch/yaml/global.yaml')
            return RIRGlobal(**cfg.net['net_g']['param'])
        
        elif name.lower() == 'add':
            from omegaconf import OmegaConf
            cfg = OmegaConf.load('./net_arch/yaml/add.yaml')
            return BaseAdd(**cfg.net['net_g']['param'])
        
        elif name.lower() == 'gate':
            from omegaconf import OmegaConf
            cfg = OmegaConf.load('./net_arch/yaml/gate.yaml')
            return BaseAddShuffle(**cfg.net['net_g']['param'])
            
        elif name.lower() == 'cat_shuffle':
            from omegaconf import OmegaConf
            cfg = OmegaConf.load('./net_arch/yaml/cat_shuffle.yaml')
            return BaseCShuffle(**cfg.net['net_g']['param'])
        
        elif name.lower() == 'uformer':
            from omegaconf import OmegaConf
            cfg = OmegaConf.load('./net_arch/yaml/uformer.yml')
            return Uformer(**cfg['net']['net_g']['param'])
        
        elif name.lower() == 'scunet':
            from omegaconf import OmegaConf
            cfg = OmegaConf.load('./net_arch/yaml/scunet.yml')
            return SCUNet(**cfg['net']['net_g']['param'])
    
    def load_ckpt(self, ckpt):
        ckpt = torch.load(ckpt, map_location=torch.device('cpu'))
        if not ckpt.get('net', False):
            return ckpt
        else:
            return ckpt['net']
    
    def __call__(self):
        return self.net
        