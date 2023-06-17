import numpy as np
import torch
from torch import nn

class ToOneHot(nn.Module):
    def forward(self, x):
        return nn.functional.one_hot(x, num_classes=256).to(torch.float)

class LabelSmoothing(nn.Module):
    def __init__(self, min_p=0.1):
        super().__init__()
        self.min_p = min_p
    
    def forward(self, label):
        return (1-self.min_p)*label + self.min_p*torch.ones_like(label)/(label.size(-1)-1)

class RandomCrop(nn.Module):
    def __init__(self, crop_length=50):
        super().__init__()
        self.crop_length = crop_length
        
    def forward(self, x):
        start_sample = np.random.randint(self.crop_length)
        end_sample = x.size(1) + start_sample - self.crop_length
        return x[:, start_sample:end_sample]
    
class SmoothBins(nn.Module):
    def __init__(self, bin_width):
        super().__init__()
        self.bin_width = bin_width
    
    def forward(self, x):
        return x + self.bin_width*torch.rand_like(x)
    
class AddNoise(nn.Module):
    def __init__(self, noise_stdev):
        super().__init__()
        self.noise_stdev = noise_stdev
    
    def forward(self, x):
        return self.noise_stdev*torch.randn_like(x) + np.sqrt(1-self.noise_stdev**2)*x
    
class RandomErasing(nn.Module):
    def __init__(self, p=0.25, max_prop=0.25):
        super().__init__()
        self.p = p
        self.max_prop = max_prop
    
    def forward(self, x):
        if np.random.rand() < self.p:
            x = x.clone()
            x_mn, x_std = torch.mean(x, dim=-1, keepdim=True), torch.std(x, dim=-1, keepdim=True)
            target_length = int(self.max_prop*np.random.rand()*x.size(-1))
            start_sample = np.random.randint(x.size(-1)-target_length)
            x[:, start_sample:start_sample+target_length] = x_std*torch.randn(1, target_length, device=x.device)+x_mn
        return x

class RandomLowPassFilter(nn.Module):
    def __init__(self, p=0.25):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if np.random.rand() < self.p:
            kernel_width = 2*np.random.randint(5)+3
            kernel = torch.ones(1, 1, kernel_width, device=x.device, dtype=x.dtype)/kernel_width
            return nn.functional.conv1d(x, kernel, padding=kernel_width//2)
        else:
            return x

class RandomHighPassFilter(nn.Module):
    def __init__(self, p=0.25):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if np.random.rand() < self.p:
            kernel_width = 2*np.random.randint(x.size(1)//100, x.size(1)//10)+3
            kernel = torch.ones(1, 1, kernel_width, device=x.device, dtype=x.dtype)/kernel_width
            return x - nn.functional.conv1d(x, kernel, padding=kernel_width//2)