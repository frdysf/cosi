from lightning.pytorch.callbacks import Callback
import torch
import wandb
import viz

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from umap.umap_ import UMAP

from typing import Union, Any
from pathlib import Path

# TODO: dict of constants to choose cmap
# based on label name

TEST_SAVE_PATH = "./src/callbacks/img"

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

        if save_path is None:
            save_path = TEST_SAVE_PATH

        save_path = Path(save_path)
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)
        if not save_path.is_dir():
            raise ValueError(f"Provided save_path {save_path} is not a directory")
        self.save_path = save_path

        self.show = show
        self.fig_kw = fig_kw
        self.ax_kw = ax_kw

        self.embeddings = []
        self.labels = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.embeddings.append(outputs["z"])
        self.labels.append(outputs["labels"])

    def on_validation_epoch_end(self, trainer, pl_module):
        # TODO: offload to async thread on CPU?

        # aggregate labels across batches
        aggregated_labels = {}
        for d in self.labels:
            for name, arr in d.items():
                if name not in aggregated_labels:
                    aggregated_labels[name] = []
                aggregated_labels[name].append(arr)

        labels = {name: torch.hstack(arr).cpu().numpy() 
                    for name, arr in aggregated_labels.items()}

        embeddings = torch.vstack(self.embeddings).cpu().numpy()

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
                show=self.show,
                save_path=self.save_path/f"{transform}_epoch={epoch}_label={name}.png",
                fig_kw=self.fig_kw,
                ax_kw=self.ax_kw,
                ax_title=f"label={name}, epoch={epoch}, transform={transform}",
            )

            # wandb.log({"key": fig}) # converts to Plotly
            # wandb.log({"key": wandb.Image(fig)}, grouping=epoch) # converts to static image

            trainer.logger.experiment.log(
                {"latent/{name}": wandb.Image(fig)},
                step=epoch,
            )

        self.embeddings.clear()
        self.labels.clear()

    def on_train_end(self, trainer, pl_module):
        # TODO: create and organised saved images into
        # subfolders according to labels
        # (unique values for label=r'label=(.+)' in path)
        # TODO: create GIF from saved images in each subfolder
        return