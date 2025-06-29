import torch
import lightning.pytorch as pl

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import logging
from pathlib import Path

import os
os.environ["HYDRA_FULL_ERROR"] = "1"  # DEBUG


OmegaConf.register_new_resolver('pow', lambda x,y: x**y)
OmegaConf.register_new_resolver('as_tuple', lambda *args: tuple(args))
OmegaConf.register_new_resolver('get_basename', lambda path: Path(path).stem)

def log_path(log_path: str, debug: bool):
    if debug:
        return "debug"
    return log_path

OmegaConf.register_new_resolver('log_path', log_path)

HYDRA_MAIN = {
    "version_base": "1.2",
    "config_path": "../config",
    "config_name": "config_train",
}

@hydra.main(**HYDRA_MAIN)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",)

    if cfg.seed:
        pl.seed_everything(cfg.seed, workers=True)

    device = cfg.features.device
    if device == "cuda":
        assert torch.cuda.is_available()
    logging.info(f"Extracting features on device: {device}")

    logging.info("Building the training pipeline...")

    # --- features ---
    feature_extractor = instantiate(cfg.features)
    
    # --- transform ---
    transform = instantiate(cfg.transform)
    logging.info(f"Instantiated transforms: {[t.__class__.__name__ for t in transform]}")

    # --- datamodule ---
    datamodule = instantiate(cfg.datamodule,
                             transform=transform,
                             feature_extractor=feature_extractor,
                             target_rate=cfg.audio.sr,
                             )
    datamodule.setup()
    datamodule.setup_feature_extractor()
    logging.info(f"Instantiated datamodule: {datamodule.__class__.__name__}")
    logging.info(f"Instantiated feature extractor: {datamodule.feature_extractor.__class__.__name__}")

    # determine feature shape to instantiate net
    if "JTFS" in cfg.features._target_:
        dummy_feature_extractor = instantiate(cfg.features)(
                                          shape=datamodule.resampled_audio_shape[-1],
                                          device="cpu",)
    else:
        dummy_feature_extractor = instantiate(cfg.features)(device="cpu")
    
    audio = datamodule.train_dataset[1]["audio"]
    resampled_audio = datamodule.resample(audio)
    feature_shape = dummy_feature_extractor(resampled_audio).shape    

    # --- net ---
    net = instantiate(cfg.net,
                      feature_shape=feature_shape)
    logging.info(f"Instantiated net: {net.__class__.__name__}")

    del dummy_feature_extractor, audio, resampled_audio, feature_shape

    # --- model ---
    model = instantiate(cfg.model,
                            net=net,
                            optimizer=instantiate(cfg.optim.optimizer),
                            lr_scheduler=instantiate(cfg.optim.scheduler),
                            lr_scheduler_params=cfg.optim.scheduler_params,)
    logging.info(f"Instantiated model: {model.__class__.__name__}")

    # --- logger ---
    logger = instantiate(cfg.logger)
    logging.info(f"Instantiated logger: {logger.__class__.__name__}")

    # --- callbacks ---
    callbacks = instantiate(cfg.callbacks)
    logging.info(f"Instantiated callbacks: {[c.__class__.__name__ for c in callbacks]}")

    # --- trainer ---
    trainer = pl.Trainer(
        deterministic=True,
        accelerator="gpu",
        devices=1,
        min_epochs=1,
        max_epochs=50,
        precision="32-true",
        callbacks=callbacks,
        logger=logger,
        profiler="simple",
        fast_dev_run=4,  # True
        # overfit_batches=1.0,
        strategy="ddp",
        default_root_dir=cfg.checkpoints_dir,
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
        # ckpt_path="",
    )
    
    # trainer.test(
    #     model=model,
    #     datamodule=datamodule,
    #     ckpt_path=None,
    # )


if __name__ == "__main__":
    main()