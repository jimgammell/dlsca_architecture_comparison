import numpy as np
import torch

class VGG(nn.Module):
    model_name = 'VGG'
    
    def __init__(self, input_shape, head_sizes, input_dropout=0.0, dropout=0.0, **kwargs):
        super().__init__()
        
        self.fe = nn.Sequential(
            nn.Dropout(input_dropout),
            nn.Conv1d(input_shape[0], 64, kernel_size=11, padding=5),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(2),
            nn.Conv1d(64, 128, kernel_size=11, padding=5),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(2),
            nn.Conv1d(128, 256, kernel_size=11, padding=5),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(2),
            nn.Conv1d(256, 512, kernel_size=11, padding=5),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(2),
            nn.Conv1d(512, 512, kernel_size=11, padding=5),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(2),
            nn.Flatten(),
            nn.Dense(32*input_shape[1], 4096),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Dense(4096, 4096),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True)
        )
        
        self.heads = nn.ModuleDict({head_name: nn.Linear(4096, head_size) for head_name, head_size in head_sizes.items()})
        
    def forward(self, x):
        x_fe = self.fe(x)
        out = {head_name: head_fn(x_fe) for head_name, head_fn in self.heads.items()}
        return out