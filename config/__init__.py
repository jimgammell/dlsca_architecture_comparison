import os
import time
import random
import numpy as np
import torch

def results_subdir(*subdir_names):
    path = os.path.join('.', 'results', *subdir_names)
    os.makedirs(path, exist_ok=True)
    return path

def set_seed(seed=None):
    if seed is None:
        seed = time.time_ns() & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed

def get_device(device=None):
    if device is None:
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        assert device in ['cuda:%d'%(dev_number) for dev_number in range(torch.cuda.device_count())]
        return device