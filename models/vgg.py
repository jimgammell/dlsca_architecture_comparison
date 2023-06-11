import numpy as np
import torch
from torch import nn
from models import layers

class VGG(nn.Module):
    model_name = 'VGG'
    
    def __init__(self, input_shape, head_sizes, input_dropout=0.0, dropout=0.0, pooling_layer='avg_pool', global_avg_pool=False, **kwargs):
        super().__init__()
        
        if pooling_layer == 'avg_pool':
            pool_constructor = lambda channels: nn.AvgPool1d(2)
        elif pooling_layer == 'max_pool_aa':
            pool_constructor = lambda channels: layers.MaxPoolAA(channels)
        
        self.fe = nn.Sequential(
            nn.Dropout(input_dropout),
            nn.Conv1d(input_shape[0], 64, kernel_size=11, padding=5),
            nn.ReLU(inplace=True),
            pool_constructor(64),
            nn.Conv1d(64, 128, kernel_size=11, padding=5),
            nn.ReLU(inplace=True),
            pool_constructor(128),
            nn.Conv1d(128, 256, kernel_size=11, padding=5),
            nn.ReLU(inplace=True),
            pool_constructor(256),
            nn.Conv1d(256, 512, kernel_size=11, padding=5),
            nn.ReLU(inplace=True),
            pool_constructor(512),
            nn.Conv1d(512, 512, kernel_size=11, padding=5),
            nn.ReLU(inplace=True),
            pool_constructor(512),
            nn.Flatten()
        )
        if global_avg_pool:
            self.pre_head = nn.Sequential(
                nn.AdaptiveAvgPool1d(4096//512),
                nn.Flatten(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
        else:
            eg_input = torch.randn(1, *input_shape)
            eg_output = self.fe(eg_input)
            self.pre_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(np.prod(eg_output.shape), 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
        
        self.heads = nn.ModuleDict({head_name: nn.Linear(4096, head_size) for head_name, head_size in head_sizes.items()})
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if m.in_channels != m.out_channels or m.out_channels != m.groups:
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x_fe = self.fe(x)
        x_ph = self.pre_head(x_fe)
        out = {head_name: head_fn(x_ph) for head_name, head_fn in self.heads.items()}
        return out