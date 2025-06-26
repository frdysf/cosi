import torch
from torch.utils.data import Dataset
import pandas as pd

from typing import Optional, Sequence, Callable, Tuple
from pathlib import Path
import os
import torchaudio

    
def int2tensor(x: int) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.uint8)  # 0 to 127

class NewOrchideaSOLDataset(Dataset):
    # TODO: implement for fresh OrchideaSOL download
    # maybe look at Haokun's examples?
    pass

class OrchideaSOLDataset(Dataset):
    '''
    PyTorch Dataset class for the OrchideaSOL dataset.
        https://forum.ircam.fr/projects/detail/orchideasol/

    The metadata CSV file is assumed to have been
    preprocessed with preprocess/sol_encode_labels.py.

    :param data_dir: Path to directory containing audio files.
    :param meta_csv: Path to CSV file containing metadata.
    :param split: 'training', 'test', 'validation' or None to use all data.
    :param transform: Sequence of transformations to preprocess the audio data.

    :returns audio: Tensor of shape (1, T) where T is the number of samples.
    :returns label_modulation: Tensor of shape (1,) containing the modulation label.
    :returns label_instrument_family: Tensor of shape (1,) containing the instrument family label.
    :returns label_pitch: Tensor of shape (1,) containing the MIDI pitch label.
    '''
    def _encode_labels(self):
        # TODO: reimplement preprocess/sol_encode_labels.py here
        pass

    def __init__(
        self, 
        data_dir: str, 
        meta_csv: str, 
        split: Optional[str], 
        transform: Optional[Sequence[Callable]] = None
    ):
        super().__init__()

        self.data_dir = data_dir

        self.data = pd.read_csv(Path(meta_csv), dtype={"midi_pitch": "UInt8"})
        self.data.dropna(subset=['midi_pitch'], inplace=True)

        if split:
            assert split in ['training', 'test', 'validation']
            self.data = self.data[self.data['subset'] == split]

        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        audio_path = Path(self.data_dir) / Path(self.data.iloc[idx]['file_name'])
        audio, sr = torchaudio.load(audio_path)

        if self.transform:
            for transform in self.transform:
                audio = transform(audio)

        label_modulation = int2tensor(self.data.iloc[idx].label_modulation)
        label_instrument_family = int2tensor(self.data.iloc[idx].label_instrument_family)
        label_pitch = int2tensor(self.data.iloc[idx].midi_pitch)

        return {
            "audio": audio,
            "labels": {
                "modulation": label_modulation,
                "instrument_family": label_instrument_family,
                "pitch": label_pitch
            }
        }
