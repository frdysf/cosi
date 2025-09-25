import torch
from torch.utils.data import DataLoader
from torch.nn import Identity
from torchaudio.transforms import Resample
import lightning as L

from typing import Optional, Sequence, Callable
from warnings import warn

from .datasets import SOLPMTDataset
from features.base import AudioFeatureExtractor

# TODO: import pytorch_mums, pytorch_nsynth and create data modules
# add data dirs and metadata files to config.paths
# NEXT: pytorch_mums, mums.py _encode_labels() and data module

# TODO: Pass collate_fn arg to DataModule to specify a custom collate function
# e.g. {'pitch': r''} where regex expression is used to (positively) filter 
# 'pitch' attr via metadata. Might need an interface to unify access to
# 'pitch' attr across datasets

class AudioDataModule(L.LightningDataModule):
    '''
    Base class for audio data modules.
    This class handles resampling and feature extraction of audio data.

    TODO: args, returns, raises
    '''
    def __init__(
        self, 
        sr: int,
        target_sr: Optional[int] = None,
    ):
        super().__init__()

        if target_sr is None:
            target_sr = sr
        self.sr = sr
        self.target_sr = target_sr
        self.resample = Resample(orig_freq=sr, new_freq=target_sr)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @property
    def resampled_audio_shape(self):
        '''
        Returns the shape of the resampled audio tensor.
        This is useful to initialize the feature extractor.

        You must call setup() before accessing this property.
        '''
        if self.train_dataset is not None:
            audio = self.train_dataset[1]["audio"]
        elif self.test_dataset is not None:
            audio = self.test_dataset[1]["audio"]
        else:
            raise RuntimeError("You must call setup() before "
                                "accessing resampled_audio_shape.")
        resampled_audio = self.resample(audio)
        return resampled_audio.shape


class SOLPMTDataModule(AudioDataModule):
    def __init__(
        self,
        data_path: str,
        csv_path: str,
        batch_size: float = 4,
        num_workers: float = 4,
        transform: Optional[Sequence[Callable]] = None,
        sr: int = 44100,
        target_sr: Optional[int] = None,
    ):
        super().__init__(sr=sr,
                         target_sr=target_sr)

        self.data_path = data_path
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    # def prepare_data(self):
    #     raise NotImplementedError

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = SOLPMTDataset(data_path=self.data_path,
                                            csv_path=self.csv_path,
                                            split='training',
                                            transform=self.transform)
        
            self.val_dataset = SOLPMTDataset(data_path=self.data_path,
                                            csv_path=self.csv_path,
                                            split='validation',
                                            transform=self.transform)
        if stage == 'test' or stage is None:
            self.test_dataset = SOLPMTDataset(data_path=self.data_path,
                                            csv_path=self.csv_path,
                                            split='test',
                                            transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )