import torch
import torch.nn as nn


class AudioFeatureExtractor(nn.Module):
    '''
    '''
    def __init__(self, sr=44100, time_avg=False, device="cpu"):
        super().__init__()
        
        self.sr = sr
        self.time_avg = time_avg
        self.device = torch.device(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = self.transform(x)
        # TODO: a) change to meanpool operation, i.e.
        # user should specify num_samples to pool over
        # b) discourage use in JTFS (use T parameter instead)
        if self.time_avg:
            x = x.mean(dim=-1)
        return x

    def to_device(self):
        self.transform = self.transform.to(self.device)