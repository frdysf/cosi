import torch
from torch.utils.data import DataLoader
from torch.nn import Identity
from torchaudio.transforms import Resample
import lightning as L

from typing import Optional, Sequence, Callable
from warnings import warn

from .datasets import OrchideaSOLDataset
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
        feature_extractor: Optional[AudioFeatureExtractor] = None,
    ):
        super().__init__()

        if feature_extractor is None:
            feature_extractor = Identity()  # pass audio through
            warn(
                "feature_extractor in DataModule is None. "
                "Using Identity() as feature extractor, "
                "which passes audio through without processing."
            )
        
        elif feature_extractor.__class__.__name__ == "partial":
                warn(
                    f"feature_extractor is a partial instantiation. "
                        "After calling `setup()` as usual, complete setup "
                        "by calling `setup_feature_extractor()`."
                    )

        self.feature_extractor = feature_extractor

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
                    
    def setup_feature_extractor(self):
        '''
        Setup the feature extractor after the data module is initialized.
        This is useful when the feature extractor requires additional parameters
        that depend on the dataset, such as the shape of the resampled audio.

        This method is called after setup().
        '''

        # TODO: refactor with similar logic in net

        if self.feature_extractor.__class__.__name__ == "partial":
            if self.feature_extractor.func.__name__ == "JTFS":
                # jtfs requires shape arg
                self.feature_extractor = self.feature_extractor(shape=self.resampled_audio_shape[-1])

        else:
            warn(
                "feature_extractor is not a partial instantiation. "
                "Ignorning call to setup_feature_extractor()."
                )

        if self.feature_extractor is not Identity:
            assert self.feature_extractor.device == torch.device("cpu"), \
                "feature_extractor must run on CPU. To use on GPU, " \
                "pass feature_extractor to net instead of datamodule."

    def on_before_batch_transfer(self, batch, dataloader_idx):
        batch["audio"] = self.resample(batch["audio"])
        batch["features"] = self.feature_extractor(batch["audio"])
        return batch


class OrchideaSOLDataModule(AudioDataModule):
    def __init__(
        self,
        data_path: str,
        csv_path: str,
        batch_size: float = 4,
        num_workers: float = 4,
        transform: Optional[Sequence[Callable]] = None,
        sr: int = 44100,
        target_sr: Optional[int] = None,
        feature_extractor: Optional[AudioFeatureExtractor] = None,
    ):
        super().__init__(sr=sr,
                         target_sr=target_sr,
                         feature_extractor=feature_extractor)

        self.data_path = data_path
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    # def prepare_data(self):
    #     raise NotImplementedError

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = OrchideaSOLDataset(data_path=self.data_path,
                                            csv_path=self.csv_path,
                                            split='training',
                                            transform=self.transform)
        
            self.val_dataset = OrchideaSOLDataset(data_path=self.data_path,
                                            csv_path=self.csv_path,
                                            split='validation',
                                            transform=self.transform)
        if stage == 'test' or stage is None:
            self.test_dataset = OrchideaSOLDataset(data_path=self.data_path,
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