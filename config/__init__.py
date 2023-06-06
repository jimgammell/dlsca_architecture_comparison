import os
import time
import copy
import json
import random
import numpy as np
import torch

def results_subdir(*subdir_names):
    path = os.path.join('.', 'results', *subdir_names)
    os.makedirs(path, exist_ok=True)
    return path

def config_dir(train=True):
    if train:
        return os.path.join('.', 'config', 'trial_settings')
    else:
        return os.path.join('.', 'config', 'htune_settings')

def get_available_configs(train=True):
    return [x.split('.')[0] for x in os.listdir(config_dir(train=train))]

def load_config(config, train=True):
    if not config in get_available_configs(train=train):
        raise Exception('Invalid config argument. Valid options: \'{}\'.'.format(
            ['\', \''.join(get_available_configs(train=train))]
        ))
    with open(os.path.join(config_dir(train=train), config+'.json'), 'r') as F:
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

def denest_dict(d, delim='-'):
    if any(delim in key for key in d.keys()):
        raise Exception('Delimiter character \'{}\' is used in one or more dictionary keys: \'{}\''.format(
            delim, '\', \''.join(list(d.keys()))))
    for key, val in copy.deepcopy(d).items():
        if type(val) == dict:
            for subkey, subval in val.items():
                d[delim.join((key, subkey))] = subval
            del d[key]
    return d

def nest_dict(d, delim='-'):
    while any(delim in key for key in d.keys()):
        for key, val in copy.deepcopy(d).items():
            if delim in key:
                outer_key, inner_key = key.split(delim, maxsplit=1)
                if not outer_key in d.keys():
                    d[outer_key] = {}
                d[outer_key][inner_key] = val
                del d[key]
    return d