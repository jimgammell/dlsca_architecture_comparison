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
        return 0.9*label + 0.1*torch.ones_like(label)/(label.size(-1)-1)

class RandomCrop(nn.Module):
    def __init__(self, crop_length=50):
        super().__init__()
        self.crop_length = crop_length
        
    def forward(self, x):
        start_sample = np.random.randint(self.crop_length)
        end_sample = x.size(1) + start_sample - self.crop_length
        x = x[:, start_sample:end_sample]
        return x
    
class AddNoise(nn.Module):
    def __init__(self, noise_stdev):
        super().__init__()
        self.noise_stdev = noise_stdev
    
    def forward(self, x):
        return noise_stdev*torch.randn_like(x) + np.sqrt(1-noise_stdev**2)*x
    
class RandomErasing(nn.Module):
    def __init__(self, p=0.25, max_prop=0.25):
        super().__init__()
        self.p = p
        self.max_prop = max_prop
    
    def forward(self, x):
        x_mn, x_std = torch.mean(x, dim=-1, keepdim=True), torch.std(x, dim=-1, keepdim=True)
        target_length = int(self.max_prop*np.random.rand()*x.size(-1))
        start_sample = np.random.randint(x.size(-1)-target_length)
        x[:, start_sample:start_sample+target_length] = x_std*torch.randn(1, target_length, device=x.device)+x_mn
        return x