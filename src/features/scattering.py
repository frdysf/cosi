import torch
import torch.nn as nn
from torch.nn import Identity
import torch.nn.functional as F

from features.base import AudioFeatureExtractor
from kymatio.torch import Scattering1D, TimeFrequencyScattering

from typing import Optional, Tuple, Union


class ScatteringFeatureExtractor(AudioFeatureExtractor):
    '''
    '''
    def __init__(
        self,
        sr: int = 44100, 
        time_avg: bool = False,
        log1p: bool = False,
        device: str = "cpu",
    ):
        if time_avg:
            raise ValueError("time_avg=True not supported for scattering features. Use T parameter instead.")
        
        super().__init__(sr=sr, time_avg=time_avg, device=device)
        self.transform = Identity()  # placeholder, replace in subclass
        self.log1p = log1p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        if self.log1p:
            x = torch.log1p(x)
        return x


class Scat1D(ScatteringFeatureExtractor):
    '''
    '''
    def __init__(
        self, 
        J: int, 
        Q: Tuple[int, int], 
        T: Optional[int],
        stride: Optional[int] = None,
        shape: Optional[Union[int, Tuple[int]]] = None, 
        sr: int = 44100, 
        time_avg: bool = False,
        log1p: bool = False,
        device: str = "cpu",
    ):  
        if time_avg:
            raise ValueError("time_avg=True is redundant for scattering features. Use T parameter instead.")

        super().__init__(sr=sr, time_avg=time_avg, log1p=log1p, device=device)

        self.transform = Scattering1D(
            shape=shape, T=T, Q=Q, J=J, stride=stride
        )
        self.to_device()
    

class JTFS(ScatteringFeatureExtractor):
    '''
    '''
    def __init__(
        self, 
        J: int,
        Q: Tuple[int, int],
        J_fr: int,
        Q_fr: int,
        T: Optional[int],
        F: int,
        stride: Optional[int] = None,
        shape: Optional[Union[int, Tuple[int]]] = None,
        format: str = "joint",
        sr: int = 44100, 
        time_avg: bool = False,
        log1p: bool = False,
        device: str = "cpu",
    ):
        if time_avg:
            raise ValueError("time_avg=True is redundant for scattering features. Use T parameter instead.")

        super().__init__(sr=sr, time_avg=time_avg, log1p=log1p, device=device)

        self.transform = TimeFrequencyScattering(
            J=J, J_fr=J_fr, shape=shape, Q=Q, T=T, Q_fr=Q_fr, F=F, stride=stride, format=format
        )
        self.to_device()