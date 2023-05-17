import numpy as np
import torch
from torch import nn

class AscadCNN(nn.Module):
    model_name='ASCAD-CNN'
    
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