import numpy as np
import torch
from torch import nn

# CnnBest from 'Study of deep learning techniques for side-channel analysis and introduction to ASCAD database' by Prouff et al.
#  Based on official implementation here: https://github.com/ANSSI-FR/ASCAD/tree/master
class ProuffCnn(nn.Module):
    model_name = 'Prouff-CNN'
    
    def __init__(self, input_shape, head_sizes, **kwargs):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_shape[0], 64, kernel_size=11, stride=1, padding=5),
            nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(64, 128, kernel_size=11, stride=1, padding=5),
            nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(128, 256, kernel_size=11, stride=1, padding=5),
            nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(256, 512, kernel_size=11, stride=1, padding=5),
            nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(512, 512, kernel_size=11, stride=1, padding=5),
            nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Flatten(),
            nn.Linear(512*(2**int(np.ceil(np.log2(input_shape[-1]))))//2**5, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        
        self.heads = nn.ModuleDict({
            head_key: nn.Linear(4096, head_size) for head_key, head_size in head_sizes.items()
        })
        
    def forward(self, x):
        x_fe = self.feature_extractor(x)
        out = {head_name: head(x_fe) for head_name, head in self.heads.items()}
        return out

# ASCAD Desync: 100 model from 'Revisiting a methodology for efficient CNN architectures in profiling attacks' by Wouters et al.
#   Based on official implementation here: https://github.com/KULeuven-COSIC/TCHES20V3_CNN_SCA/tree/master
#   Tweaks:
#     Changed kernel sizes to odd numbers so that things look cleaner
class WoutersNet(nn.Module):
    model_name = 'WoutersNet'
    
    def __init__(
        self,
        input_shape, # Dimensions of the input trace. Should have form (# channels, # samples). # channels is probably 1.
        head_sizes, # Dictionary mapping head name to # logits in head. Should match names in the dataset labels.
        simplify_kernel=True, # Whether to use odd number kernel sizes, for aesthetic purposes.
        precise_bn=False, # Whether to use precise batchnorm (accurately compute stats right before testing, instead of relying on EMA).
        **kwargs
    ):
        super().__init__()
        
        if precise_bn:
            bn_kwargs = {'momentum': None, 'eps': 1e-3}
        else:
            bn_kwargs = {'momentum': 0.01, 'eps': 1e-3}
        
        self.feature_extractor = nn.Sequential(*([
            nn.AvgPool1d(2),
            nn.Conv1d(
                input_shape[0], 64,
                kernel_size=51 if simplify_kernel else 50,
                stride=1,
                padding=25 if simplify_kernel else 0
            )
        ] + ([] if simplify_kernel else [nn.ConstantPad1d((24, 25), 0)]) + [
            nn.SELU(),
            nn.BatchNorm1d(64, **bn_kwargs),
            nn.AvgPool1d(50),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.SELU(),
            nn.BatchNorm1d(128, **bn_kwargs),
            nn.AvgPool1d(2)
        ]))
        
        eg_input = torch.randn(1, *input_shape)
        self.pre_head = nn.Sequential(
            nn.Linear(self.feature_extractor(eg_input).view(-1).shape[0], 20),
            nn.SELU(),
            nn.Linear(20, 20),
            nn.SELU(),
            nn.Linear(20, 20),
            nn.SELU()
        )
        
        self.heads = nn.ModuleDict({
            head_key: nn.Linear(20, head_size) for head_key, head_size in head_sizes.items()
        })
        
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='linear')
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, np.prod(x.shape[1:]))
        x = self.pre_head(x)
        x = {head_name: head(x) for head_name, head in self.heads.items()}
        return x

class KimNet(nn.Module):
    def __init__(self, input_shape, head_sizes, dropout_rate=0.5):
        super().__init__()
        
        class VGGBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                
                self.block = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.BatchNorm1d(2*channels),
                    nn.ReLU(),
                    nn.MaxPool1d(2)
                )
            
            def forward(self, x):
                return self.block(x)
        
        self.feature_extractor = nn.Sequential(
            VGGBlock(input_shape[0], 8),
            VGGBlock(8, 16),
            VGGBlock(16, 32),
            VGGBlock(32, 64),
            VGGBlock(64, 128),
            VGGBlock(128, 256),
            VGGBlock(256, 256),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.pre_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(dropout_rate)
        )
        self.heads = nn.ModuleDict({
            head_key: nn.Linear(256, head_size) for head_key, head_size in head_sizes.items()
        })
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pre_head(x)
        x = {head_name: head(x) for x in self.heads.items()}
        return x
        