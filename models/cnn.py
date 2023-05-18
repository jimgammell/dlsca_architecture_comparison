import numpy as np
import torch
from torch import nn

class Stem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=1, padding=3),
            nn.AvgPool1d(2)
        )
        
    def forward(self, x):
        return self.stem(x)

class IsotropicBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, groups=channels, kernel_size=7, bias=False, stride=1, padding=3),
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels*4, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(channels*4, channels, kernel_size=1)
        )
        
    def forward(self, x):
        return (self.block(x) + x) / np.sqrt(2)
    
class DownsampleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(channels)
        )
        
    def forward(self, x):
        return self.block(x)

class ConvNext(nn.Module):
    def __init__(self, input_shape, head_sizes):
        super().__init__()
        
        self.stem = Stem(input_shape[0], 64)
        self.blocks = nn.Sequential(
            DownsampleBlock(128),
            DownsampleBlock(256)
        )
        self.pre_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU()
        )
        self.heads = nn.ModuleDict({
            head_key: nn.Linear(256, head_size) for head_key, head_size in head_sizes.items()
        })
        
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pre_head(x)
        x = {head_name: head(x) for head_name, head in self.heads.items()}
        return x