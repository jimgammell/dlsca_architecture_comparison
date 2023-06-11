from collections import OrderedDict
import numpy as np
import torch
from torch import nn

class ConvMixer(nn.Module):
    model_name = 'ConvMixer'
    
    def __init__(
        self,
        input_shape,
        depth=8,
        patch_length=50,
        spatial_kernel_size=11,
        hidden_channels=256,
        spatial_skip_conn=True,
        channel_skip_conn=False,
        activation_constructor=None,
        norm_constructor=type('BatchNorm', (nn.BatchNorm1d,), {}),
        **kwargs
    ):
        super().__init__()
        
        if type(activation_constructor) == str:
            activation_constructor = getattr(nn, activation_constructor)
        elif activation_constructor is None:
            activation_constructor = nn.SELU if norm_constructor is None else nn.GELU
        if type(norm_constructor) == str:
            norm_constructor = getattr(nn, norm_constructor)
        if norm_constructor is None:
            norm_constructor = nn.Identity
        
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
                self.skip_conn = spatial_skip_conn
                
            def forward(self, x):
                out = self.model(x)
                if self.skip_conn:
                    out = (out + x) / np.sqrt(2)
                return out
        
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
                self.skip_conn = channel_skip_conn
                
            def forward(self, x):
                out = self.model(x)
                if self.skip_conn:
                    out = (out + x) / np.sqrt(2)
                return out
        
        self.patch_embedding = nn.Sequential(OrderedDict([
            ('conv', nn.Conv1d(
                in_channels=input_shape[0], out_channels=hidden_channels, kernel_size=patch_length,
                stride=patch_length, padding=(patch_length-(input_shape[1]%patch_length))%patch_length
            ))
        ]))
        
        self.pre_mixer = nn.Sequential(OrderedDict([
            ('act', activation_constructor()),
            ('norm', norm_constructor(hidden_channels))
        ]))
        
        self.mixer = nn.Sequential(OrderedDict([
            ('layer_{}'.format(layer_idx),
             nn.Sequential(OrderedDict([('spatial_mixer', SpatialMixer()), ('channel_mixer', ChannelMixer())]))
            ) for layer_idx in range(depth)
        ]))
        
        self.head = nn.Sequential(OrderedDict([
            ('pool', nn.AdaptiveAvgPool1d(1)),
            ('flatten', nn.Flatten()),
            ('dense', nn.Linear(hidden_channels, 256))
        ]))
        
        def init_weights(mod):
            if isinstance(mod, (nn.Linear, nn.Conv1d)):
                if norm_constructor is not None:
                    nn.init.kaiming_uniform_(mod.weight, nonlinearity='relu')
                else:
                    nn.init.kaiming_uniform_(mod.weight, nonlinearity='linear')
                mod.bias.data.zero_()
        self.apply(init_weights)
        
    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.pre_mixer(x)
        x = self.mixer(x)
        x = self.head(x)
        return x

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
    print('ConvMixer: {} -> {}'.format(eg_input.shape, eg_output.shape))
    
if __name__ == '__main__':
    main()