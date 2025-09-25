import torch
from torch.utils.data import Dataset
import pandas as pd

from typing import Optional, Sequence, Callable, Tuple
from pathlib import Path
import os
import torchaudio

    
def uint8tensor(x: int) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.uint8)  # 0 to 127

class OrchideaSOLDataset(Dataset):
    # TODO: placeholder, use Haokun's code
    pass

class SOLPMTDataset(Dataset):
    '''
    PyTorch Dataset class for the SOL-PMT dataset.

    The metadata CSV file is assumed to have been
    preprocessed with preprocess/sol_encode_labels.py.

    :param data_path: Path to directory containing audio files.
    :param csv_path: Path to CSV file containing metadata.
    :param split: 'training', 'test', 'validation' or None to use all data.
    :param transform: Sequence of transformations to preprocess the audio data.

    :returns audio: Tensor of shape (1, T) where T is the number of samples.
    :returns label_modulation: Tensor of shape (1,) containing the modulation label.
    :returns label_instrument_family: Tensor of shape (1,) containing the instrument family label.
    :returns label_pitch: Tensor of shape (1,) containing the MIDI pitch label.
    '''
    def __init__(
        self, 
        data_path: str, 
        csv_path: str, 
        split: Optional[str], 
        transform: Optional[Sequence[Callable]] = None
    ):
        super().__init__()

        self.data_path = data_path

        self.data = pd.read_csv(Path(csv_path), dtype={"midi_pitch": "UInt8"})
        self.data.dropna(subset=['midi_pitch'], inplace=True)

        if split:
            assert split in ['training', 'test', 'validation']
            self.data = self.data[self.data['subset'] == split]

        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        audio_path = Path(self.data_path) / Path(self.data.iloc[idx]['file_name'])
        audio, sr = torchaudio.load(audio_path)

        if self.transform is not None:
            for transform in self.transform:
                audio = transform(audio)

        label_modulation = uint8tensor(self.data.iloc[idx].label_modulation)
        label_instrument_family = uint8tensor(self.data.iloc[idx].label_instrument_family)
        label_pitch = uint8tensor(self.data.iloc[idx].midi_pitch)

        return {
            "audio": audio,
            "labels": {
                "modulation": label_modulation,
                "instrument_family": label_instrument_family,
                "pitch": label_pitch,
            }
        }
