import torch
import torch.nn as nn
import torch.nn.functional as F

from features.base import AudioFeatureExtractor
from torchaudio.transforms import MFCC, MelSpectrogram, AmplitudeToDB

from typing import Optional


class MFCC(AudioFeatureExtractor):
    '''
    '''
    def __init__(
        self,
        n_mfcc: int,
        log_mels: bool = True,
        sr: int = 44100,
        time_avg: bool = False,
        device: str = "cpu",
        melkwargs: Optional[dict] = None,
    ):
        super().__init__(sr=sr, time_avg=time_avg, device=device)

        self.transform = MFCC(
            sample_rate = sr,
            n_mfcc = n_mfcc,
            log_mels = log_mels,
            melkwargs = melkwargs,
        )
        self.to_device()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transform(x)
        if self.time_avg:
            x = x.mean(dim=-1)
        var = x.var(dim=-1)
        return x, var


class LogMelSpec(AudioFeatureExtractor):
    '''
    '''
    def __init__(
        self, 
        n_fft: int, 
        hop_length: int, 
        sr: int = 44100, 
        time_avg: bool = False,
        device: str = "cpu",
    ):
        super().__init__(sr=sr, time_avg=time_avg, device=device)

        self.transform = nn.Sequential(
            MelSpectrogram(
                sample_rate = sr,
                n_fft = n_fft,
                hop_length = hop_length
            ),
            AmplitudeToDB(),
        )
        self.to_device()


class MelSpec(AudioFeatureExtractor):
    '''
    '''
    def __init__(
        self, 
        n_fft: int, 
        hop_length: int, 
        sr: int = 44100, 
        time_avg: bool = False,
        device: str = "cpu",
    ):
        super().__init__(sr=sr, time_avg=time_avg, device=device)

        self.transform = nn.Sequential(
            MelSpectrogram(
                sample_rate = sr,
                n_fft = n_fft,
                hop_length = hop_length
            ),
        )
        self.to_device()