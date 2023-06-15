# adapted from https://github.com/davda54/sam/blob/main/example/model/wide_res_net.py and https://github.com/owruby/shake-shake_pytorch/blob/master/models/shakeshake.py

from collections import OrderedDict
import numpy as np
import torch
from torch import nn

class ShakeShake(torch.autograd.Function):
    @staticmethod
    def forward(self, x1, x2, training=True):
        if training:
            alpha = torch.randn(x1.size(0), 1, 1, device=x1.device, dtype=x1.dtype).expand_as(x1)
        else:
            alpha = 0.5
        return alpha * x1 + (1-alpha) * x2
    
    @staticmethod
    def backward(self, grad_output):
        beta = torch.rand(grad_output.size(0), 1, 1, device=grad_output.device, dtype=grad_output.dtype).expand_as(grad_output)
        beta = torch.autograd.Variable(beta)
        return beta * grad_output, (1-beta)*grad_output, None

class BasicUnit(nn.Module):
    def __init__(self, channels, dropout, kernel_size=3, shake_shake=False):
        super().__init__()
        
        def get_block():
            return nn.Sequential(OrderedDict([
                ('norm_0', nn.BatchNorm1d(channels)),
                ('act_0', nn.ReLU(inplace=True)),
                ('conv_0', nn.Conv1d(channels, channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)),
                ('norm_1', nn.BatchNorm1d(channels)),
                ('act_1', nn.ReLU(inplace=True)),
                ('dropout_0', nn.Dropout(dropout, inplace=True)),
                ('conv_1', nn.Conv1d(channels, channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False))
            ]))
        if shake_shake:
            self.block_1, self.block_2 = get_block(), get_block()
        else:
            self.block = get_block()
        self.shake_shake = shake_shake
        
    def forward(self, x):
        if self.shake_shake:
            x_resid1, x_resid2 = self.block_1(x), self.block_2(x)
            x_resid = ShakeShake.apply(x_resid1, x_resid2, self.training)
        else:
            x_resid = self.block(x)
        out = x + x_resid
        return out

class DownsampleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dropout, kernel_size=3, shake_shake=False):
        super().__init__()
        
        self.norm_act = nn.Sequential(OrderedDict([
            ('norm_0', nn.BatchNorm1d(in_channels)),
            ('act_0', nn.ReLU(inplace=True))
        ]))
        def get_block():
            return nn.Sequential(OrderedDict([
                ('conv_0', nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)),
                ('norm_0', nn.BatchNorm1d(out_channels)),
                ('act_0', nn.ReLU(inplace=True)),
                ('dropout_0', nn.Dropout(dropout, inplace=True)),
                ('conv_1', nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False))
            ]))
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        if shake_shake:
            self.block_1, self.block_2 = get_block(), get_block()
        else:
            self.block = get_block()
        self.shake_shake = shake_shake
        
    def forward(self, x):
        x = self.norm_act(x)
        if self.shake_shake:
            x_resid1, x_resid2 = self.block_1(x), self.block_2(x)
            x_resid = ShakeShake.apply(x_resid1, x_resid2, self.training)
        else:
            x_resid = self.block(x)
        out = self.downsample(x) + x_resid
        return out

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, depth, dropout, kernel_size=3, shake_shake=False):
        super().__init__()
        
        self.block = nn.Sequential(
            DownsampleUnit(in_channels, out_channels, stride, dropout, kernel_size=kernel_size, shake_shake=shake_shake),
            *(BasicUnit(out_channels, dropout, kernel_size=kernel_size, shake_shake=shake_shake) for _ in range(depth))
        )
        
    def forward(self, x):
        return self.block(x)
        
class WideResNet(nn.Module):
    model_name = 'WideResNet'
    
    def __init__(self, input_shape, depth=16, dropout=0.0, width_factor=1, kernel_size=11, shake_shake=False, **kwargs):
        super().__init__()
        
        self.filters = [16, 1*16*width_factor, 2*16*width_factor, 4*16*width_factor]
        self.block_depth = (depth-4) // (3*2)
        
        self.model = nn.Sequential(OrderedDict([
            ('conv_0', nn.Conv1d(input_shape[0], self.filters[0], kernel_size=kernel_size, stride=1, padding=1, bias=False)),
            ('block_0', Block(self.filters[0], self.filters[1], 1, self.block_depth, dropout, shake_shake=shake_shake, kernel_size=kernel_size)),
            ('block_1', Block(self.filters[1], self.filters[2], 2, self.block_depth, dropout, shake_shake=shake_shake, kernel_size=kernel_size)),
            ('block_2', Block(self.filters[2], self.filters[3], 2, self.block_depth, dropout, shake_shake=shake_shake, kernel_size=kernel_size)),
            ('norm_0', nn.BatchNorm1d(self.filters[3])),
            ('act_0', nn.ReLU(inplace=True)),
            ('pool_0', nn.AdaptiveAvgPool1d(1)),
            ('flatten_0', nn.Flatten()),
            ('classification_0', nn.Linear(self.filters[3], 256))
        ]))
        self._initialize()
        
    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.zero_()
                m.bias.data.zero_()
    
    def forward(self, x):
        return self.model(x)
