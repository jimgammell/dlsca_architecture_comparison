import numpy as np
import torch
from torch import nn

class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        )
    
    def forward(self, x):
        x_avg = self.fc(x.mean(dim=-1))
        x_max = self.fc(x.max(dim=-1))
        att = torch.sigmoid(x_avg + x_max)
        att = out.view(x.size(0), x.size(1), 1)
        out = att * x
        return out

class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.conv = nn.Conv1d(2, 1, kernel_size=11, padding=5, bias=False)
        
    def forward(self, x):
        x_avg = x.mean(dim=-1, keepdims=True)
        x_max, _ = x.max(dim=-1, keepdims=True)
        att = self.conv(torch.cat([x_avg, x_max], dim=1))
        att = torch.sigmoid(att)
        out = att * x
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.residual_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=11),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=11),
            nn.ReLU(inplace=True),
            ChannelAttention(out_channels),
            SpatialAttention(out_channels)
        )
        
    def forward(self, x):
        return x + self.residual_block(x)

class PoolingBlock(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.AvgPool1d(2),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

class CBAM(nn.Module):
    def __init__(self, input_shape, head_sizes, widths=[128, 256, 512], fc_neurons=4096):
        super().__init__()
        
        modules = []
        widths = [input_shape[0]] + widths
        for wi, wo in widths[:-1], widths[1:]:
            modules.extend([ConvBlock(wi, wo), PoolingBlock()])
        modules.extend([
            nn.Flatten(),
            nn.Linear(input_shape[1]*128, fc_neurons),
            nn.ReLU(inplace=True)
        ])
        self.fe = nn.Sequential(*modules)
        self.heads = nn.ModuleDict({
            head_key: nn.Linear(fc_neurons, head_size) for head_key, head_size in head_sizes.items()
        })
        
    def forward(self, x):
        x_fe = self.fe(x)
        out = {head_key: head(x_fe) for head_key, head in self.heads}
        return out