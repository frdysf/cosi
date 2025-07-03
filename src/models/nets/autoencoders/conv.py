import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Identity

from typing import Optional, Tuple
from warnings import warn

from .base import BaseAutoencoder


class ConvAutoencoder(BaseAutoencoder):
    '''
    Convolutional autoencoder for JTFS features.
    Baseline model.

    :param feature_shape: Shape (C, Nf, Nt, Npath) of input features.
    :param latent_dim: Latent dimension of the embedding space.
    :param activation_fn: Activation function.
    '''
    def __init__(
        self,
        sr: int, 
        input_shape: Tuple[int, ...],
        latent_dim: int,
        activation_fn: nn.Module = nn.ReLU(),
        feature_extractor: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.sr = sr
        self.latent_dim = latent_dim

        # TODO: refactor with similar logic in datamodule

        if feature_extractor is None:
            feature_extractor = Identity()  # pass audio through
            warn(
                "feature_extractor in net is None. "
                "Using Identity() as feature extractor, "
                "which passes audio through without processing."
            )
        elif feature_extractor.__class__.__name__ == "partial":
            if feature_extractor.func.__name__ == "JTFS":
                # jtfs requires shape arg
                feature_extractor = feature_extractor(shape=input_shape[-1])

        if feature_extractor is not Identity:
            assert feature_extractor.device == torch.device("cuda"), \
                "feature_extractor must be CUDA-enabled. To use on CPU, " \
                "pass feature_extractor to datamodule instead of net."

        self.feature_extractor = feature_extractor

        dummy_input = torch.zeros(input_shape)
        feature_shape = self.feature_extractor(dummy_input).shape

        self.encoder = Encoder(feature_shape, latent_dim, activation_fn)
        self.decoder = Decoder(feature_shape, latent_dim, activation_fn)


class Encoder(nn.Module):
    '''
    :param feature_shape: Shape of input features.
    :param latent_dim: Latent dimension of the embedding space.
    :param activation_fn: Activation function.
    '''
    def __init__(
        self,
        feature_shape: Tuple[int, int, int, int],
        latent_dim: int,
        activation_fn: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        C, Npath, Nf, Nt = feature_shape

        self.net = nn.Sequential(
                nn.LayerNorm([Nf, Nt, Npath, C], 
                             elementwise_affine=False, 
                             bias=False),  # per-sample normalization
                *[ConvBlock(activation_fn=activation_fn,
                                in_channels=_in, 
                                out_channels=_out, 
                                kernel_size=_k,
                                stride=_s,
                                groups=_g)
                    for _in, _out, _k, _s, _g in [(Nf, 48, (3, 8, 1), (1, 4, 6), 4),
                                              (48, 12, (5, 5, 1), (1, 2, 4), 3)]],
                nn.Flatten(),
                nn.Linear(in_features=360, out_features=72),
                activation_fn,
                nn.Linear(in_features=72, out_features=latent_dim),
                activation_fn,
            )

    def forward(self, x):
        if x.ndim == 4:  # missing batch dimension
            x = x.unsqueeze(0)  # [1, Nf, Nt, Npath, C]
        x = x.permute(0, 3, 4, 2, 1)  # [B, C, Npath, Nf, Nt]
        return self.net(x)


class Decoder(nn.Module):
    '''
    :param target_shape: Shape of reconstructed output.
    :param latent_dim: Latent dimension of the embedding space.
    :param activation_fn: Activation function.
    '''
    def __init__(
        self,
        target_shape: Tuple[int, int, int, int],
        latent_dim: int,
        activation_fn: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        C, Npath, Nf, Nt = target_shape

        # 1) build sub-encoder just up through the conv layers
        # to deduce output shape before flattening
        dummy_encoder = Encoder(target_shape, latent_dim)
        # extract only the LayerNorm + ConvBlocks:
        conv_only = nn.Sequential(*list(dummy_encoder.net.children())[: 1 + 2])
        # 1 layer for LayerNorm + 2 ConvBlocks

        with torch.no_grad():
            dummy_input = torch.zeros(1, Nf, Nt, Npath).unsqueeze(-1)
            conv_out = conv_only(dummy_input)
        # will be shape [1, C, D, H, W]
        _, Cc, Dc, Hc, Wc = conv_out.shape
        self._conv_shape = (Cc, Dc, Hc, Wc)

        # 2) now build decoder
        self.net = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=72),
            activation_fn,
            nn.Linear(in_features=72, out_features=Cc * Dc * Hc * Wc),
            activation_fn,
            nn.Unflatten(1, self._conv_shape),
            *[DeconvBlock(activation_fn=activation_fn,
                                in_channels=_in, 
                                out_channels=_out, 
                                kernel_size=_k,
                                stride=_s,
                                groups=_g,
                                output_padding=_p)
                for _in, _out, _k, _s, _g, _p in [(12, 48, (5, 5, 1), (1, 2, 4), 3, 0),
                                            (48, Nf, (3, 8, 1), (1, 4, 6), 4, (0, 1, 0))]],
            # learnable to approximate inverse transformation
            nn.LayerNorm([Nf, Nt, Npath, C], 
                elementwise_affine=True, 
                bias=True),
        )

    def forward(self, z):
        x = self.net(z)
        return x.permute(0, 4, 3, 1, 2)  # [B, Nf, Nt, Npath, C]
    
    
class ConvBlock(nn.Module):
    def __init__(
        self,
        activation_fn: nn.Module,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int],
        stride: Tuple[int],
        groups: int = 1,
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, groups=groups),
            nn.BatchNorm3d(out_channels),
            activation_fn,
            )
    
    def forward(self, x):
        return self.layers(x)
    

class DeconvBlock(nn.Module):
    def __init__(
        self,
        activation_fn: nn.Module,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int],
        stride: Tuple[int],
        groups: int = 1,
        output_padding: Tuple[int, ...] = 0,
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride, groups=groups,
                               output_padding=output_padding),
            nn.BatchNorm3d(out_channels),
            activation_fn,
            )

    def forward(self, x):
        return self.layers(x)
