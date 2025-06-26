import torch
import torch.nn as nn
import torch.nn.functional as F

from features.base import AudioFeatureExtractor
from kymatio.torch import Scattering1D, TimeFrequencyScattering

from typing import Optional, Tuple, Union


class Scat1D(AudioFeatureExtractor):
    '''
    '''
    def __init__(
        self, 
        J: int, 
        Q: Tuple[int, int], 
        T: Optional[int],
        shape: Optional[Union[int, Tuple[int]]] = None, 
        sr: int = 44100, 
        time_avg: bool = False,
        device: str = "cpu",
    ):
        super().__init__(sr=sr, time_avg=time_avg, device=device)

        self.transform = Scattering1D(
            shape=shape, T=T, Q=Q, J=J
        )
        self.to_device()
    

class JTFS(AudioFeatureExtractor):
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
        shape: Optional[Union[int, Tuple[int]]] = None,
        format: str = "joint",
        sr: int = 44100, 
        time_avg: bool = False,
        device: str = "cpu",
    ):
        super().__init__(sr=sr, time_avg=time_avg, device=device)

        self.transform = TimeFrequencyScattering(
            J=J, J_fr=J_fr, shape=shape, Q=Q, T=T, Q_fr=Q_fr, F=F, format=format
        )
        self.to_device()
