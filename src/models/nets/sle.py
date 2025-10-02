import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence, Optional, Callable, Union
from models.utils import get_activation_fn

Activation = Callable[..., nn.Module]

class SharedLocalEncoder(nn.Module):
    def __init__(self, 
                 conv_channels: Sequence[int], 
                 fc_dims: Sequence[int], 
                 output_dim: int,
                 output_activation: Optional[Union[Activation, str]] = None,
                 use_batchnorm: bool = False,
                 pooling: str = 'flatten',
        ):
        """
        Simple CNN-based local feature encoder. Cascades 1x1 convolutions and fully connected layers.

        Args:
            conv_channels (Sequence[int]): Output channels for the 3 1x1 conv layers.
            fc_dims (Sequence[int]): Hidden dims for the first two FC layers.
            output_dim (int): Dimension of the final embedding.
            output_activation (torch.nn.Module, optional): Activation function at output layer.
            use_batch_norm (bool): If True, applies BatchNorm after each learned layer.
            pooling (str): Pooling method to aggregate spatial features. Options: 'gap' for global average pooling or 'flatten'.
        """
        super().__init__()
        assert pooling in ('gap', 'flatten'), "pooling must be 'gap' or 'flatten'"
        self.pooling = pooling

        conv_layers = []
        for i, out_ch in enumerate(conv_channels):
            if i == 0:
                conv_layers.append(nn.LazyConv2d(out_ch, kernel_size=1))
            else:
                conv_layers.append(nn.Conv2d(conv_channels[i-1], out_ch, kernel_size=1))
            if use_batchnorm:
                conv_layers.append(nn.BatchNorm2d(out_ch))
            conv_layers.append(nn.ReLU(inplace=True))
        self.convs = nn.Sequential(*conv_layers)

        fc_layers = []
        for i, fc_out in enumerate(fc_dims):
            if i == 0:
                fc_layers.append(nn.LazyLinear(fc_out))
            else:
                fc_layers.append(nn.Linear(fc_dims[i-1], fc_out))
            if use_batchnorm:
                fc_layers.append(nn.BatchNorm1d(fc_out))
            fc_layers.append(nn.ReLU(inplace=True))

        # final layer
        fc_layers.append(nn.Linear(fc_dims[-1], output_dim))
        if use_batchnorm:
            fc_layers.append(nn.BatchNorm1d(output_dim))

        if output_activation:
            if isinstance(output_activation, str):
                output_activation = get_activation_fn(output_activation)
            fc_layers.append(output_activation())

        self.fcs = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.convs(x)            # (B, C, H, W)
        if self.pooling == 'flatten':
            x = x.flatten(start_dim=1)  # flatten all but batch dim
        elif self.pooling == 'gap':
            x = x.mean(dim=[2, 3])       # global average pooling
        x = self.fcs(x)
        return x