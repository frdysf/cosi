from lightning.pytorch.callbacks import Callback
import torch
import viz 

# TODO: callback to save/collect plot at each validation step
# (i.e. per epoch) then on training end, create and save/display GIF

class VisualizeLatents(Callback):
    
    def __init__(
            self, 
            # transform, 
            # n_components,
            # save_path,
            # show,
            # fig_kw,
            # ax_kw,
            # ax_title,
        ):
        super().__init__()
        # self.transform = transform
        # self.n_components = n_components

        self.embeddings = []
        self.labels = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.embeddings.append(outputs["z"])
        self.labels.append(outputs["labels"])

    def on_validation_epoch_end(self, trainer, pl_module):
        # TODO: offload to async thread on CPU?

        embeddings = torch.vstack(self.embeddings).cpu().numpy()
        print(embeddings.shape)  # DEBUG

        # aggregate (across arbitrary number of) labels
        # labels = torch.dstack(
        #     [torch.vstack(label) for label in self.labels]
        # ).flatten(1,2).cpu().numpy()

        print(self.labels)  # DEBUG

        # transformed_embeddings = viz.transform(
        #     embeddings,
        #     transform=self.transform,
        #     n_components=self.n_components,
        # )

        # for d in range(labels.dim(0)):
        # plot transformed embeddings

        # pass embeddings and labels to plot function
        # log plot as wandb artefact
        # wandb = pl_module.logger.experiment
        # wandb.log({"embeddings": wandb.Image(plot)})
        # OR pl_module.log()

        self.embeddings.clear()
        self.labels.clear()