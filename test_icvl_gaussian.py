# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/10/30 11:48:06
# @File    :   test_icvl_gaussian.py
# @Contact   :   lianghao@whu.edu.cn

import os
import glob
import argparse
import os.path as osp

from utils import *
from tqdm import tqdm
from net_arch import *
from model import Model
from thop import profile
from rich.table import Table
from rich.console import Console
from scipy.io import loadmat, savemat

parser = argparse.ArgumentParser(
    prog='test_icvl_gaussian',
    description='HSI Denoising',
    epilog='Copyright(r) 2023'
)

parser.add_argument('--arch', default='sst')
parser.add_argument('--ckpt', default='./')
parser.add_argument('--save_dir', default=None)
parser.add_argument('--device', default='cpu')
parser.add_argument('--index', default=0, type=int)

args = parser.parse_args()

@memory_time_decorator
def test_inference(net, input):
    """test the runtime and memory on inference
    Args:
        net (nn.Module): net
        input (tensor): (1 c h w)
    """
    return net(input)

def get_complexity(net, input):
    macs, params = profile(net, (input, ), verbose=False)
    return {'params': params / 1e6, 'FLOPs': (macs * 2) / 1e9}

def test(net, mat_name, mat_dir, device, args):
    if args.save_dir:
        save_path = os.path.join(args.save_dir, f"{mat_name}.mat")
        print(f'will save on {save_path}')
        
    mats = [path for path in glob.glob(osp.join(mat_dir, '*.mat'))]
    num = len(mats)
    
    resource = {'memory': 0., 'time': 0.}
    metric = {'psnr': 0., 'ssim': 0., 'sam': 0.}
    
    save_data = {'lq': [], 'hq': [], 'denoised': []}
    
    net.eval()
    with torch.no_grad():
        for i in tqdm(range(num), desc='inference'):
            data = loadmat(mats[i])
            '''
            中心裁剪
            '''
            pch = 256
            h, w, c = data['gt'].shape
            ind_h = (h - pch) // 2
            ind_w = (w - pch) // 2
            data['gt'] = data['gt'][ind_h:ind_h + pch, 
                                    ind_w:ind_w + pch
                                    ]
            data['input'] = data['input'][ind_h:ind_h + pch, 
                                    ind_w:ind_w + pch
                                    ]


            hq = np.float32(data['gt'])
            lq = hsi2tensor(np.float32(data['input']))
            lq_tensor = lq.to(device)
            
            res = test_inference(net, lq_tensor)
            
            resource['memory'] += res['memory']
            resource['time'] += res['time']
            
            denoised = tensor2hsi(res['denoised'])
            res = cal_metric(hq, denoised)     
            
            metric['psnr'] += res['psnr']
            metric['ssim'] += res['ssim']    
            metric['sam'] += res['sam']
            
            save_data['hq'].append(hq)
            save_data['lq'].append(np.float32(data['input']))
            save_data['denoised'].append(denoised)

    
    if args.save_dir:
        for key in save_data.keys():
            save_data[key] = np.array(save_data[key])
        savemat(save_path, save_data)
        
        with open(os.path.join(args.save_dir, "res.txt"), 'a+') as file:
            file.seek(0)
            file.write('\n' + f'======================= {mat_name} ==========================\n')
            file.write(f"psnr = {metric['psnr'] / num : .2f}, ssim = {metric['ssim'] / num : .4f}, sam = {metric['sam'] / num : .2f}\n")
            print('write all')
        
    complexity = {'params': 0., 'FLOPs': 0.}
    with torch.no_grad():
        for i in tqdm(range(num), desc='complexity'):
            lq = hsi2tensor(np.float32(data['input']))
            lq_tensor = lq.to(device)
            
            res = get_complexity(net, lq_tensor)
            
            complexity['params'] += res['params']
            complexity['FLOPs'] += res['FLOPs']
    
    return {'complexity': complexity , 'metric': metric, 'resource': resource, 'num': num}

def main(net, test_dataset, device, args):
    if args.save_dir and not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    console = Console()
    tabel = Table(show_header=True, header_style='bold magenta', title=f'{args.arch}')
    tabel.add_column('Name', width=16, justify='center')
    tabel.add_column('PSNR (↑)', width=16, justify='center')
    tabel.add_column('SSIM (↑)', width=16, justify='center')
    tabel.add_column('SAM (↓)', width=16, justify='center')
    tabel.add_column('Memory (MB)', width=16, justify='center')
    tabel.add_column('RunTime(s)', width=16, justify='center')
    tabel.add_column('Param(M)', width=16, justify='center')
    tabel.add_column('GFLOPs', width=16, justify='center')
    
    for name, mat_dir in test_dataset.items():
        info = test(net, name, mat_dir, device, args)
        num = info['num']
        tabel.add_row(name, f"{info['metric']['psnr'] / num :.2f}", 
                      f"{info['metric']['ssim'] / num : .4f}", f"{info['metric']['sam'] / num :.2f}",
                      f"{info['resource']['memory'] / num :.2f}", f"{info['resource']['time'] / num :.2f}",
                      f"{info['complexity']['params'] / num :.2f}", f"{info['complexity']['FLOPs'] / num :.2f}")
            
    console.print(tabel)
    
    

if __name__ == '__main__':
    
    test_dataset = test_dataset = {
        'sigma-10': '/home/lianghao/data/icvl_test/512_10',
        'sigma-30': '/home/lianghao/data/icvl_test/512_30',
        'sigma-50': '/home/lianghao/data/icvl_test/512_50',
        'sigma-70': '/home/lianghao/data/icvl_test/512_70',
        'sigma-10-70': '/home/lianghao/data/icvl_test/512_10_70'  , 
        'cave-10-70': '/home/lianghao/data/cave_test/gaussian_blind',
        'cave-50': '/home/lianghao/data/cave_test/gaussian_50'
    }


    if args.device == 'cpu':
        device = torch.device(args.device)
    else:
        device = torch.device(args.device, index=args.index)
        
    net = Model(arch_name=args.arch, device=device, ckpt=args.ckpt)()
    
    main(net, test_dataset, device, args=args)