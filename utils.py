# -*- encoding: utf-8 -*-
# @Author  :   Liang Hao 
# @Time    :   2023/10/30 11:35:15
# @File    :   utils.py
# @Contact   :   lianghao@whu.edu.cn


import os
import gc
import time
import torch
import psutil
import numpy as np
import imgvision as iv

from functools import wraps

def tensor2hsi(tensor):
    return np.transpose(tensor.data.cpu().numpy().squeeze(), (1, 2, 0))

def hsi2tensor(hsi, mode='HWC'):
    if mode.upper() == 'HWC':
        return torch.from_numpy(np.transpose(hsi[np.newaxis, ...], (0, 3, 1, 2)))
    elif mode.upper() == 'CHW':
        return torch.from_numpy(hsi[np.newaxis, ...])
    else:
        raise f"hsi2tensor-mode is error"

def cal_metric(hq, lq):
    metric = iv.spectra_metric(hq, lq)
    return {
        'psnr': metric.PSNR(),
        'ssim': metric.SSIM(),
        'sam': metric.SAM()
    }


def memory_time_decorator(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        # get the information for this process
        process = psutil.Process(os.getpid())
        gc.collect()
        # initial memory
        mem_before = process.memory_info().rss / 1024 / 1024
        
        # initial time
        start_time = time.time()
        
        res = function(*args, **kwargs)
        
        # elapsed time
        elapsed_time = time.time() - start_time
        
        # end memory
        gc.collect()
        mem_after = process.memory_info().rss / 1024 / 1024
        
        mem_usage = mem_after - mem_before
        
        return {'memory': mem_usage, 'time': elapsed_time, 'denoised': res}
    
    return wrapper
