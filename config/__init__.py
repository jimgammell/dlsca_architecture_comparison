import os
import time
import json
import random
import numpy as np
import torch

def results_subdir(*subdir_names):
    path = os.path.join('.', 'results', *subdir_names)
    os.makedirs(path, exist_ok=True)
    return path

def config_dir():
    path = os.path.join('.', 'config', 'trial_settings')
    return path

def get_available_configs():
    return [x.split('.')[0] for x in os.listdir(config_dir())]

def load_config(config):
    if not config in get_available_configs():
        raise Exception('Invalid config argument. Valid options: \'{}\'.'.format(
            ['\', \''.join(get_available_configs())]
        ))
    with open(os.path.join(config_dir(), config+'.json'), 'r') as F:
        settings = json.load(F)
    return settings

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