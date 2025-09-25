import torch
import lightning.pytorch as pl

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pathlib import Path

import os
os.environ["HYDRA_FULL_ERROR"] = "1"  # DEBUG


def log_dir(dir_name: str, debug: bool):
    assert isinstance(debug, bool)
    if debug:
        return "debug"
    return dir_name

OmegaConf.register_new_resolver('log_dir', log_dir)
OmegaConf.register_new_resolver('pow', lambda x,y: x**y)
OmegaConf.register_new_resolver('div', lambda x,y: x/y)
OmegaConf.register_new_resolver('as_tuple', lambda *args: tuple(args))
OmegaConf.register_new_resolver('get_basename', lambda path: Path(path).stem)

HYDRA_MAIN = {
    "version_base": "1.2",
    "config_path": "../config",
    "config_name": "config_train",
}

@hydra.main(**HYDRA_MAIN)
def main(cfg: DictConfig) -> None:
    # TODO: back to logging instead of print lol
    if cfg.seed:
        pl.seed_everything(cfg.seed, workers=True)

    device = cfg.features.device
    if device == "cuda":
        assert torch.cuda.is_available()
    print(f"[train] Set to extract features on device: {device}")

    print("[train] Building the training pipeline...")

    # --- features ---
    feature_extractor = instantiate(cfg.features)
    
    # --- transform ---
    transform = instantiate(cfg.transform)
    print(f"[train] Instantiated transforms: {[t.__class__.__name__ for t in transform]}")

    # --- datamodule ---
    datamodule = instantiate(cfg.datamodule,
                             transform=transform,
                             target_sr=cfg.audio.target_sr,
                             feature_extractor=feature_extractor if device == "cpu" else None,
                             )
    datamodule.setup()
    # datamodule.setup_feature_extractor()
    print(f"[train] Instantiated datamodule: {datamodule.__class__.__name__}")
    print(f"[train] Set to resample audio to {cfg.audio.target_sr} Hz. Resampled audio shape: {datamodule.resampled_audio_shape}")

    # # determine input shape to instantiate net if using datamodule.feature_extractor
    # if "JTFS" in cfg.features._target_:
    #     dummy_feature_extractor = instantiate(cfg.features)(
    #                                       shape=datamodule.resampled_audio_shape[-1],
    #                                       device="cpu",)
    # else:
    #     dummy_feature_extractor = instantiate(cfg.features)(device="cpu")
    # resampled_audio = datamodule.resample(audio)
    # feature_shape = dummy_feature_extractor(resampled_audio).shape   
    
    # determine input shape to instantiate net if not using datamodule.feature_extractor
    audio = datamodule.train_dataset[1]["audio"]
    input_shape = datamodule.resampled_audio_shape

    # --- net ---
    net = instantiate(cfg.net,
                      input_shape=input_shape,
                      feature_extractor=feature_extractor if device == "cuda" else None,
                      )
    print(f"[train] Instantiated net: {net.__class__.__name__}")
    print(f"[train] Instantiated feature extractor (inside net): {net.feature_extractor.__class__.__name__}")

    del audio, input_shape

    # --- model ---
    model = instantiate(cfg.model,
                            net=net,
                            optimizer=instantiate(cfg.optim.optimizer),
                            lr_scheduler=instantiate(cfg.optim.scheduler),
                            lr_scheduler_params=cfg.optim.scheduler_params,)
    print(f"[train] Instantiated model: {model.__class__.__name__}")

    # --- logger ---
    logger = pl.loggers.WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.run,
        save_dir=cfg.wandb.save_dir,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        log_model=True,
    )

    # --- callbacks ---
    callbacks = instantiate(cfg.callbacks)
    print(f"[train] Instantiated callbacks: {[c.__class__.__name__ for c in callbacks]}")

    trainer = instantiate(cfg.trainer,
                          logger=logger,
                          callbacks=callbacks,
                          )

    # trainer.fit(
    #     model=model,
    #     datamodule=datamodule,
    #     # ckpt_path="",
    # )
    
    # trainer.test(
    #     model=model,
    #     datamodule=datamodule,
    #     ckpt_path=None,
    # )

if __name__ == "__main__":
    main()