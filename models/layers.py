import numpy as np
import torch
from torch import nn

# Based on the official implementation: https://github.com/adobe/antialiased-cnns/blob/master/antialiased_cnns/blurpool.py
class BlurPool(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=5, stride=2, pad_off=0):
        super().__init__()
        
        self.filt_size = filt_size
        if filt_size != 5:
            raise NotImplementedError
        self.channels = channels
        self.stride = stride
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size-1) / 2), int(np.ceil(1. * (filt_size-1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.off = int((self.stride - 1) / 2.)
        
        blur_kernel = torch.Tensor([1., 4., 6., 4., 1.])
        blur_kernel /= torch.sum(blur_kernel)
        blur_kernel = blur_kernel[None, None, :].repeat((self.channels, 1, 1))
        self.register_buffer('blur_kernel', blur_kernel)
        if pad_type in ['refl', 'reflect']:
            self.pad = nn.ReflectionPad1d(self.pad_sizes)
        elif pad_type in ['repl', 'replace']:
            self.pad = nn.ReplicationPad1d(self.pad_sizes)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad1d(self.pad_sizes)
        
    def forward(self, x):
        return nn.functional.conv1d(self.pad(x), self.blur_kernel, stride=self.stride, groups=x.shape[1])

class MaxPoolAA(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=1),
            BlurPool(*args, **kwargs)
        )
        
    def forward(self, x):
        return self.pool(x)

# Based on this implementation: https://github.com/deepmind/deepmind-research/blob/master/nfnets/base.py
class RescaledNonlinearity(nn.Module):
    def __init__(self, nonlinearity, nonlinearity_kwargs={}):
        super().__init__()
        
        if type(nonlinearity) == str:
            nonlinearity = getattr(nn, nonlinearity)
        
        self.rescalar = {
            nn.Identity: 1.0,
            nn.GELU: 1.7015043497085571,
            nn.LeakyReLU: 1.70590341091156,
            nn.ReLU: 1.7139588594436646,
            nn.SELU: 1.0008515119552612,
            nn.SiLU: 1.7881293296813965
        }[nonlinearity]
        self.nonlinearity = nonlinearity(**nonlinearity_kwargs)
        
    def forward(self, x):
        return self.rescalar * self.nonlinearity(x)
    
    def __repr__(self):
        return '%f * {}'%(self.nonlinearity)

class Rescale(nn.Module):
    def __init__(self, scalar):
        super().__init__()
        
        self.scalar = scalar
    
    def forward(self, x):
        return self.scalar * x
    
    def __repr__(self):
        return 'Rescale(scalar={})'.format(self.scalar)