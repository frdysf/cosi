from lightning.pytorch.callbacks import Callback
import viz

from lightning.pytorch.utilities import rank_zero_only
import torch
import torch.distributed as dist

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from umap.umap_ import UMAP

from typing import Union, Any
from pathlib import Path

from lightning.pytorch.loggers import WandbLogger
import wandb


CMAP = {  # assign colormaps for preknown labels
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
            every_n_epoch: int = 1,
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

        self.every_n_epoch = every_n_epoch

    @rank_zero_only
    def visualize(self, trainer, embeddings, labels):
        embeddings = embeddings.cpu().numpy()
        labels = {key: arr.cpu().numpy() for key, arr in labels.items()}

        reduced_embeddings = viz.reduce_embeddings(
                embeddings,
                transform=self.transform,
                n_components=self.n_components,
            )

        try:
            transform = self.transform.__class__.__name__
        except AttributeError:
            transform = str(self.transform)
        epoch = trainer.current_epoch

        for key in labels.keys():
            fig = viz.plot_reduced_embeddings(
                reduced_embeddings,
                n_components=self.n_components,
                labels=labels[key],
                cmap=CMAP.get(key, None),
                show=self.show,
                save_path=self.save_path/f"{transform}_epoch={epoch}_label={key}.png",
                fig_kw=self.fig_kw,
                ax_kw=self.ax_kw,
                ax_title=f"label={key}, epoch={epoch}, transform={transform}",
            )

            trainer.logger.experiment.log(
                {f"latent/{key}": wandb.Image(fig)  # plotly begone
                    if isinstance(trainer.logger, WandbLogger) else fig,
                "epoch": epoch},
            )

    def on_validation_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epoch != 0:
            return

        local_embeddings = pl_module.stacked_embeddings
        local_labels = pl_module.stacked_labels

        # gather embeddings and labels from all processes
        if dist.is_initialized() and (world_size := distr.get_world_size()) > 1:
            gathered_embeddings = [torch.zeros_like(local_embeddings) for _ in range(world_size)]
            gathered_labels = [None for _ in range(world_size)]

            dist.gather(local_embeddings, gathered_embeddings)
            dist.gather_object(local_labels, gathered_labels)

            local_embeddings = torch.cat(gathered_embeddings, dim=0)
            local_labels = {key: torch.cat([label[key] 
                       for label in gathered_labels])
                         for key in gathered_labels[0].keys()}

        self.visualize(trainer, local_embeddings, local_labels)

    def on_train_end(self, trainer, pl_module):
        # TODO: create and organised saved images into
        # subfolders according to labels
        # (unique values for label=r'label=(.+)' in path)
        # TODO: create GIF from saved images in each subfolder
        # TODO: log to wandb
        return