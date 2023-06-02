from collections import OrderedDict
import numpy as np
import torch
from torch import nn

class ConvMixer(nn.Module):
    model_name = 'ConvMixer'
    
    def __init__(
        self,
        input_shape,
        head_sizes,
        depth=8,
        patch_length=50,
        spatial_kernel_size=3,
        hidden_channels=256,
        activation_constructor=type('GELU', (nn.GELU,), {}),
        norm_constructor=type('BatchNorm', (nn.BatchNorm1d,), {})
    ):
        super().__init__()
        
        class SpatialMixer(nn.Module):
            def __init__(self):
                super().__init__()
                
                self.model = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv1d(
                        in_channels=hidden_channels, out_channels=hidden_channels, groups=hidden_channels,
                        kernel_size=spatial_kernel_size, stride=1, bias=True, padding=spatial_kernel_size//2
                    )),
                    ('act', activation_constructor()),
                    ('norm', norm_constructor(hidden_channels))
                ]))
                
            def forward(self, x):
                return x + self.model(x)
        
        class ChannelMixer(nn.Module):
            def __init__(self):
                super().__init__()
                
                self.model = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv1d(
                        in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=1
                    )),
                    ('act', activation_constructor()),
                    ('norm', norm_constructor(hidden_channels))
                ]))
                
            def forward(self, x):
                return self.model(x)
        
        
        
        patch_embedding = nn.Sequential(OrderedDict([
            ('conv', nn.Conv1d(
                in_channels=input_shape[0], out_channels=hidden_channels, kernel_size=patch_length,
                stride=patch_length, padding=(patch_length-(input_shape[1]%patch_length))%patch_length
            ))
        ]))
        
        pre_mixer = nn.Sequential(OrderedDict([
            ('act', activation_constructor()),
            ('norm', norm_constructor(hidden_channels))
        ]))
        
        mixer = nn.Sequential(OrderedDict([
            ('layer_{}'.format(layer_idx),
             nn.Sequential(OrderedDict([('spatial_mixer', SpatialMixer()), ('channel_mixer', ChannelMixer())]))
            ) for layer_idx in range(depth)
        ]))
        
        pooling = nn.Sequential(OrderedDict([
            ('pool', nn.AdaptiveAvgPool1d(1)),
            ('flatten', nn.Flatten())
        ]))
        
        self.feature_extractor = nn.Sequential(OrderedDict([
            ('patch_embedding', patch_embedding),
            ('pre_mixer', pre_mixer),
            ('mixer', mixer),
            ('pooling', pooling)
        ]))
        
        self.heads = nn.ModuleDict({
            head_key: nn.Linear(hidden_channels, head_size) for head_key, head_size in head_sizes.items()
        })
        
    def forward(self, x):
        features = self.feature_extractor(x)
        out = {head_name: head(features) for head_name, head in self.heads.items()}
        return out

def main():
    print('Testing ConvMixer code.')
    batch_size = 32
    input_shape = (1, 1500)
    head_sizes = {'byte_2': 256}
    spatial_kernel_size = 11
    model = ConvMixer(input_shape, head_sizes, spatial_kernel_size=spatial_kernel_size)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('ConvMixer model with {} parameters:'.format(n_params))
    print(model)
    eg_input = torch.randn(batch_size, *input_shape)
    eg_output = model(eg_input)
    print('ConvMixer: {} -> {}'.format(eg_input.shape, {name: output.shape for name, output in eg_output.items()}))
    
if __name__ == '__main__':
    main()