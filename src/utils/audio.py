import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from warnings import warn


class Normalize(nn.Module):
    '''
    Normalize audio to range [-1, 1].
    '''
    def __init__(self, eps: float = 1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        max_val = max(torch.abs(audio).max(), self.eps)
        return audio / max_val
    

class EqualizeDuration(nn.Module):
    '''
    Pad or truncate audio to a fixed duration.
    If both dur_secs and dur_samples are set, dur_samples will take precedence.
    '''
    def __init__(
        self, 
        sr: int = 44100, 
        dur_secs: float = 1.0, 
        dur_samples: Optional[int] = None
    ):
        super().__init__()
        self.sr = sr
        self.dur_secs = dur_secs
        self.dur_samples = dur_samples

        if dur_secs and dur_samples:
            warn("Both dur_secs and dur_samples are set for "
                      "equalize_duration. dur_samples will take precedence.")

        if self.dur_samples is not None:
            self.target_samples = self.dur_samples
        else:
            self.target_samples = int(self.sr * self.dur_secs)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        n_samples = audio.shape[1]

        if n_samples > self.target_samples:
            audio = audio[:,:self.target_samples]
        else:
            audio = F.pad(audio, (0, self.target_samples - n_samples))

        assert audio.shape[1] == self.target_samples
        return audio