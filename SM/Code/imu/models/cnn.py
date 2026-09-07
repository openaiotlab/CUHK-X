import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ConvolutionalBlocks(nn.Module):
    def __init__(self, 
                in_channels, 
                sample_length,
                out_channels=[32, 64, 128],  
                kernel_size=3, 
                stride=1, 
                padding=1,
                pool_padding=0, 
                pool_size=2,
                **kwargs):
        """
        1D-Convolutional Network
        """
        super(ConvolutionalBlocks, self).__init__()
        self.name = 'conv_layers_1d'
        self.num_layers = len(out_channels)
        self.out_size = self._compute_out_size(out_channels[-1], sample_length, self.num_layers, padding, kernel_size, stride, pool_padding, pool_size=pool_size)
        
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels[0], kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels[0]),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels[1]),
            nn.ReLU(),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels[2]),
            nn.ReLU(),
        )

    @staticmethod
    def _compute_out_size(num_channels, sample_length, num_layers, padding, kernel_size, stride, pool_padding, pool_size):
        conv_out_size = sample_length
        for _ in range(num_layers):
            conv_out_size = int((conv_out_size + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
        return int(num_channels * conv_out_size)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return x


class SupervisedCNN1D(nn.Module):
    def __init__(
        self,
        in_channels, 
        sample_length, 
        out_size,
        out_channels=[32, 64, 128],  
        kernel_size=3, 
        stride=1, 
        padding=1,
        pool_padding=0, 
        pool_size=2,
        modality='acc',
        device='cuda',
        **kwargs
    ):
        super().__init__()
        self.conv_layers = ConvolutionalBlocks(
            in_channels, 
            sample_length, 
            out_channels=out_channels, 
            kernel_size=kernel_size,
            stride=stride, 
            padding=padding, 
            pool_padding=pool_padding, 
            pool_size=pool_size
        )
        self.classifier = nn.Linear(self.conv_layers.out_size, out_size)
        self.modality = modality
        self.device = device
        self.to(device)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)

def kaiming_init_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    