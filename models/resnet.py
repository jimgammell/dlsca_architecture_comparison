import numpy as np
import torch
from torch import nn

class StemBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(out_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels//2, out_channels//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels//2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        for _, module in self.named_modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
    def forward(self, x):
        return self.block(x)

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_seblock=False, reduction_ratio=0.25, id_init=False):
        super().__init__()
        
        self.resid_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//4, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels//4, out_channels//4, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False),
            nn.BatchNorm1d(out_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels//4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            *([SqueezeExciteBlock(out_channels, reduction_ratio)] if use_seblock else []),
            nn.ReLU(inplace=True)
        )
        
        if stride != 1 or in_channels != out_channels:
            self.sc_block = nn.Sequential(
                nn.Identity() if stride == 1 else nn.AvgPool1d(2, stride=stride, ceil_mode=True, count_include_pad=False),
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.sc_block = nn.Sequential()
        
        self.output_act = nn.ReLU(inplace=True)
        
        for _, module in self.named_modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        if id_init:
            nn.init.zeros_([module.weight for _, module in self.named_modules() if isinstance(module, nn.BatchNorm1d)][-1])
        
    def forward(self, x):
        x_resid = self.resid_block(x)
        x_sc = self.sc_block(x)
        out = self.output_act(x_resid + x_sc)
        return out

class SqueezeExciteBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=0.25):
        super().__init__()
        
        reduced_channels = int(channels * reduction_ratio)
        self.block = nn.Sequential(
            nn.Conv1d(channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduced_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x_se = x.mean(2, keepdim=True)
        x_se = self.block(x_se)
        out = x * x_se
        return out

class ResNet(nn.Module):
    model_name = "ResNet"
    
    def __init__(
        self, 
        input_shape,
        head_sizes,
        initial_channels=16,
        blocks_per_stage=[1, 1, 1, 1, 1, 1],
        kernel_size=3,
        use_seblocks=False,
        reduction_ratio=0.25,
        id_init=False,
        dropout_ratio=0.0,
        **kwargs
    ):
        super().__init__()
        
        self.stem = StemBlock(input_shape[0], initial_channels)
        
        block_kwargs = {
            'kernel_size': kernel_size, 'use_seblock': use_seblocks, 'reduction_ratio': reduction_ratio, 'id_init': id_init
        }
        self.stages = nn.ModuleList([
            nn.Sequential(
                BottleneckBlock(initial_channels*2**sidx, initial_channels*2**(sidx+1), stride=2, **block_kwargs),
                *[BottleneckBlock(initial_channels*2**(sidx+1), initial_channels*2**(sidx+1), stride=1, **block_kwargs) for _ in range(blocks_per_stage[sidx]-1)]
            ) for sidx in range(len(blocks_per_stage))
        ])
        
        self.pre_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout_ratio),
            nn.Linear(initial_channels*2**len(blocks_per_stage), initial_channels*2**len(blocks_per_stage))
        )
        
        self.heads = nn.ModuleDict({
            head_name: nn.Linear(initial_channels*2**len(blocks_per_stage), head_size) for head_name, head_size in head_sizes.items()
        })
        
        for _, module in self.named_modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
    def forward(self, x):
        x = nn.functional.interpolate(x, size=(512,))
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.pre_head(x)
        out = {head_name: head(x) for head_name, head in self.heads.items()}
        return out
        