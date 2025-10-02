import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from typing import Optional, Sequence, Callable, Tuple
from pathlib import Path
import os
import torchaudio

    
def uint8tensor(x: int) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.uint8)  # 0 to 127

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
        data_dir: str, 
        data_csv: str, 
        split: Optional[str], 
        transform: Optional[Sequence[Callable]] = None,
    ):
        super().__init__()

        self.data_dir = data_dir

        self.df = pd.read_csv(Path(data_csv), dtype={"midi_pitch": "UInt8"})
        self.df.dropna(subset=['midi_pitch'], inplace=True)

        if split:
            assert split in ['training', 'test', 'validation']
            self.df = self.df[self.df['subset'] == split]

        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        audio_path = Path(self.data_dir) / Path(self.df.iloc[idx]['file_name'])
        audio, sr = torchaudio.load(audio_path)

        if self.transform is not None:
            for transform in self.transform:
                audio = transform(audio)

        label_modulation = uint8tensor(self.df.iloc[idx].label_modulation)
        label_instrument_family = uint8tensor(self.df.iloc[idx].label_instrument_family)
        label_pitch = uint8tensor(self.df.iloc[idx].midi_pitch)

        return {
            "audio": audio,
            "labels": {
                "modulation": label_modulation,
                "instrument_family": label_instrument_family,
                "pitch": label_pitch,
            }
        }
