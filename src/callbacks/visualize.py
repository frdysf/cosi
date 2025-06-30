from lightning.pytorch.callbacks import Callback
import torch
import wandb
import viz

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from umap.umap_ import UMAP

from typing import Union, Any
from pathlib import Path


CMAP = {
    "pitch": "viridis",
    "modulation": "tab20",
    "instrument_family": "tab20",
}

class VisualizeLatents(Callback):
    def __init__(
            self, 
            transform: Union[PCA, UMAP, TSNE, Isomap, Any], 
            n_components: int,
            save_path: str,
            show: bool,
            fig_kw: dict,
            ax_kw: dict,
        ):
        super().__init__()
        self.transform = transform
        self.n_components = n_components

        save_path = Path(save_path)
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)
        if not save_path.is_dir():
            raise ValueError(f"Provided save_path {save_path} is not a directory")
        self.save_path = save_path

        self.show = show
        self.fig_kw = fig_kw
        self.ax_kw = ax_kw

    def on_validation_end(self, trainer, pl_module):
        # TODO: async with other callbacks?
        embeddings = pl_module.stacked_embeddings
        labels = pl_module.stacked_labels

        transformed_embeddings = viz.transform(
            embeddings,
            transform=self.transform,
            n_components=self.n_components,
        )

        try:
            transform = self.transform.__class__.__name__
        except AttributeError:
            transform = str(self.transform)
        epoch = trainer.current_epoch
        for name in labels.keys():
            fig = viz.plot(
                transformed_embeddings,
                n_components=self.n_components,
                labels=labels[name],
                cmap=CMAP[name],
                show=self.show,
                save_path=self.save_path/f"{transform}_epoch={epoch}_label={name}.png",
                fig_kw=self.fig_kw,
                ax_kw=self.ax_kw,
                ax_title=f"label={name}, epoch={epoch}, transform={transform}",
            )

            trainer.logger.experiment.log(
                {f"latent/{name}": wandb.Image(fig),
                 "epoch": epoch},
            )

    def on_train_end(self, trainer, pl_module):
        # TODO: create and organised saved images into
        # subfolders according to labels
        # (unique values for label=r'label=(.+)' in path)
        # TODO: create GIF from saved images in each subfolder
        return