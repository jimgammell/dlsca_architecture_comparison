import numpy as np
import torch
from torch import nn

class ResBlockBase(nn.Module):
    def __init__(self, survival_rate=1.0):
        super().__init__()
        
        self.survival_rate = survival_rate
    
    # Based on TorchVision implementation here:
    #   https://pytorch.org/vision/main/_modules/torchvision/ops/stochastic_depth.html#stochastic_depth
    def apply_stochastic_depth(self, x_rc):
        if self.survival_rate < 1.0:
            size = [x_rc.size(0)] + [1]*(x.dim()-1)
            noise = torch.empty(size, dtype=x_rc.dtype, device=x_rc.device).bernoulli_(self.survival_rate)
            if self.survival_rate > 0.0:
                noise.div_(survival_rate)
            return x_rc * noise
        return x_rc
    
    def forward(self, x):
        x_sc = self.shortcut_connection(x)
        x_rc = self.residual_connection(x)
        
        assert x_sc.size(0) == x_rc.size(0)
        assert x_sc.size(1) == x_rc.size(1)
        if x_sc.size(2) < x_rc.size(2):
            pad_len = x_rc.size(2) - x_sc.size(2)
            x_sc = nn.functional.pad(x_sc, (pad_len//2, (pad_len//2)+(1 if pad_len%2!=0 else 0)))
        elif x_sc.size(2) > x_rc.size(2):
            drop_len = x_sc.size(2) - x_rc.size(2)
            x_sc = x_sc[:, :, drop_len//2:-((drop_len//2)+(1 if drop_len%2!=0 else 0))]
        else:
            pass
        
        if self.training:
            x_rc = self.apply_stochastic_depth(x_rc)
        
        out = (x_sc + x_rc) / np.sqrt(2)
        return out

class ResNetBlock(ResBlockBase):
    def __init__(self, in_channels, out_channels, 
                 downsample=False, kernel_size=3,
                 activation_class=None, norm_class=None,
                 survival_rate=1.0, id_init=False
                ):
        super().__init__(survival_rate=survival_rate)
        
        residual_modules = []
        if norm_class is not None:
            residual_modules.append(norm_class(in_channels))
        if activation_class is not None:
            residual_modules.append(activation_class())
        residual_modules.append(nn.Conv1d(
            in_channels, out_channels//4, kernel_size=1, bias=norm_class is None
        ))
        if norm_class is not None:
            residual_modules.append(norm_class(out_channels//4))
        if activation_class is not None:
            residual_modules.append(activation_class())
        residual_modules.append(nn.Conv1d(
            out_channels//4, out_channels//4, kernel_size=kernel_size, bias=norm_class is None,
            stride=2 if downsample else 1, padding=kernel_size//2
        ))
        if norm_class is not None:
            residual_modules.append(norm_class(out_channels//4))
        if activation_class is not None:
            residual_modules.append(activation_class())
        residual_modules.append(nn.Conv1d(
            out_channels//4, out_channels, kernel_size=1, bias=norm_class is None
        ))
        
        if id_init:
            assert norm_class is not None
            norm_layers = [mod for mod in residual_modules if isinstance(mod, norm_class)]
            assert norm_layers[-1].weight is not None
            norm_layers[-1].weight.zero_()
        
        shortcut_modules = []
        if downsample:
            shortcut_modules.append(nn.AvgPool1d(2))
        if in_channels != out_channels:
            shortcut_modules.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        
        self.residual_connection = nn.Sequential(*residual_modules)
        self.shortcut_connection = nn.Sequential(*shortcut_modules)
        
class ConvNextBlock(ResBlockBase):
    def __init__(self, in_channels, out_channels,
                 kernel_size=7, activation_constructor=None,
                 norm_constructor=None, survival_rate=1.0):
        super().__init__(survival_rate=survival_rate)
        
        residual_modules = []
        residual_modules.append(nn.Conv1d(
            in_channels, in_channels, groups=in_channels, kernel_size=kernel_size, bias=norm_constructor is None
        ))
        if norm_constructor is not None:
            residual_modules.append(norm_constructor())
        residual_modules.append(nn.Conv1d(in_channels, out_channels*4, kernel_size=1))
        if activation_constructor is not None:
            residual_modules.append(activation_constructor())
        residual_modules.append(nn.Conv1d(out_channels*4, out_channels, kernel_size=1))
        
        shortcut_modules = []
        if (in_channels != out_channels):
            shortcut_modules.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
            
        self.residual_connection = nn.Sequential(*residual_modules)
        self.shortcut_connection = nn.Sequential(*shortcut_modules)
        
class ResNet(nn.Module):
    model_name = 'ResNet'
    
    def __init__(
        self, 
        input_shape,
        head_sizes,
        stage_blocks=[3, 4, 6, 3],
        base_channels=64, kernel_size=3,
        activation_class=type('ReLUip', (nn.ReLU,), {'inplace': True}), 
        norm_class=type('PreciceBatchNorm1d', (nn.BatchNorm1d,), {'momentum': None, 'track_running_stats': True}),
        survival_rate=1.0, id_init=False
    ):
        super().__init__()
        block_kwargs = {
            'kernel_size': kernel_size, 'activation_class': activation_class,
            'norm_class': norm_class, 'survival_rate': survival_rate, 'id_init': id_init
        }
        
        self.stem = nn.Sequential(
            nn.Conv1d(input_shape[0], base_channels, kernel_size=7, stride=2, padding=3),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.stages = nn.ModuleList([])
        for sidx, n in enumerate(stage_blocks):
            stage_modules = []
            stage_modules.append(ResNetBlock(
                base_channels*2**sidx, base_channels*2**(sidx+1), downsample=True, **block_kwargs
            ))
            for _ in range(n-1):
                stage_modules.append(ResNetBlock(
                    base_channels*2**(sidx+1), base_channels*2**(sidx+1), downsample=False, **block_kwargs
                ))
            stage = nn.Sequential(*stage_modules)
            self.stages.append(stage)
            
        self.pooling_layer = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        self.heads = nn.ModuleDict({
            head_key: nn.Linear(base_channels*2**len(stage_blocks), head_size) for head_key, head_size in head_sizes.items()
        })
        
    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.pooling_layer(x)
        out = {head_name: head(x) for head_name, head in self.heads.items()}
        return out
    