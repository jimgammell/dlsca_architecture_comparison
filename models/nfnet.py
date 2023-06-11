import numpy as np
import torch
from torch import nn

from models import layers

class TransitionBlock(nn.Module):
    def __init__(
        self,
        channels,
        nonlinearity,
        beta,
        alpha=0.2
    ):
        super().__init__()
        
        self.preprocess = nn.Sequential(
            layers.Rescale(1/beta),
            nonlinearity()
        )
        self.residual = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nonlinearity(),
            nn.Conv1d(channels, channels, groups=channels//128, kernel_size=3, stride=2, padding=1),
            nonlinearity(),
            nn.Conv1d(channels, channels, groups=channels//128, kernel_size=3, padding=1),
            nonlinearity(),
            nn.Conv1d(channels, 2*channels, kernel_size=1),
            layers.SEBlock(),
            layers.Rescale(alpha)
        )
        self.skip = nn.Sequential(
            nn.AvgPool1d(2),
            nn.Conv1d(channels, 2*channels, kernel_size=1)
        )
        
    def forward(self, x):
        x = self.preprocess(x)
        x_residual = self.residual(x)
        x_skip = self.skip(x)
        out = x_residual + x_skip
        return out

class NonTransitionBlock(nn.Module):
    def __init__(
        self,
        channels,
        nonlinearity,
        beta,
        alpha=0.2
    ):
        super().__init__()
        
        self.residual = nn.Sequential(
            layers.Rescale(1/beta),
            nn.Conv1d(channels, channels//2, kernel_size=1),
            nonlinearity(),
            nn.Conv1d(channels//2, channels//2, groups=channels//128, kernel_size=3, padding=1),
            nonlinearity(),
            nn.Conv1d(channels//2, channels//2, groups=channels//128, kernel_size=3, padding=1),
            nonlinearity(),
            nn.Conv1d(channels//2, channels, kernel_size=1),
            layers.SEBlock(),
            layers.Rescale(alpha)
        )
        
    def forward(self, x):
        x_residual = self.residual(x)
        out = x + x_residual
        return out

class NFNet(nn.Module):
    def __init__(
        self,
        input_shape,
        head_sizes,
        initial_channels=256,
        stage_depths=[1, 2, 6, 3],
        dropout_rate=0.2,
        nonlinearity=layers.RescaledNonlinearity(nn.GELU)
    ):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv1d(input_shape[0], initial_channels//8, kernel_size=3, stride=2, padding=1),
            nonlinearity(),
            nn.Conv1d(initial_channels//8, initial_channels//4, kernel_size=3, padding=1),
            nonlinearity(),
            nn.Conv1d(initial_channels//4, initial_channels//2, kernel_size=3, padding=1),
            nonlinearity(),
            nn.Conv1d(initial_channels//2, initial_channels, kernel_size=3, stride=2, padding=1)
        )