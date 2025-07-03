import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torchmetrics

from typing import Dict, Optional, Any


class AutoencoderModule(L.LightningModule):
    '''
    '''
    def __init__(
        self,
        net: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        lr_scheduler_params: Optional[Dict[str, Any]] = None,
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net", "loss_fn"])
        self.net = net
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params or {}
        self.loss_weights = loss_weights or {"recon": 1.0}

        assert all(key in self.loss_weights for key in ["recon"]), \
            "Loss weights must include 'recon' key."

        self.embeddings = []
        self.labels = []

    def forward(self, x):
        x, x_hat, z = self.net(x)
        return z

    def compute_loss(self, x, x_hat, z) -> Dict[str, torch.Tensor]:
        # standard reconstruction loss
        recon_loss = self.loss_fn(x_hat, x)

        # extendable loss dictionary
        losses = {"recon": recon_loss}

        # example: add contrastive or latent regularization loss
        # if hasattr(self, 'contrastive_loss'):
        #     losses["contrastive"] = self.contrastive_loss(z)

        # combine losses with weights
        total_loss = sum(self.loss_weights[k] * losses[k] for k in losses)
        losses["total"] = total_loss
        return losses
    
    def model_step(self, batch, batch_idx):
        x = batch["features"]
        labels = batch["labels"]
        x, x_hat, z = self.net(x)
        losses = self.compute_loss(x, x_hat, z)
        return x, x_hat, z, labels, losses

    def training_step(self, batch, batch_idx):
        x, x_hat, z, labels, losses = self.model_step(batch, batch_idx)
        self.log_dict({f"train/{k}": v for k, v in losses.items()}, 
                      on_step=False, on_epoch=True, sync_dist=True)
        return {"loss": losses["total"], "z": z, "x_hat": x_hat, "labels": labels}

    def validation_step(self, batch, batch_idx):
        x, x_hat, z, labels, losses = self.model_step(batch, batch_idx)
        self.log_dict({f"val/{k}": v for k, v in losses.items()}, 
                      on_step=False, on_epoch=True, sync_dist=True)
        return {"loss": losses["total"], "z": z, "x_hat": x_hat, "labels": labels}
    
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        self.embeddings.append(outputs["z"])
        self.labels.append(outputs["labels"])
    
    def on_validation_epoch_end(self):
        aggregated_labels = {}  # gather labels from across batches
        for d in self.labels:
            for name, arr in d.items():
                if name not in aggregated_labels:
                    aggregated_labels[name] = []
                aggregated_labels[name].append(arr)

        self.stacked_labels = {name: torch.hstack(arr).detach()
                    for name, arr in aggregated_labels.items()}

        self.stacked_embeddings = torch.vstack(self.embeddings).detach()

    def on_validation_end(self):
        self.embeddings.clear()
        self.labels.clear()
        del self.stacked_embeddings
        del self.stacked_labels

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())

        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    **self.lr_scheduler_params,
                }
            }
        
        return {"optimizer": optimizer}


class VAEModule(AutoencoderModule):  # TODO
    '''
    '''
    def __init__(
        self,
        net: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__(net, loss_fn, optimizer, lr_scheduler, loss_weights)
        self.loss_weights = loss_weights or {"recon": 0.5, "kl": 0.5}
        
        assert all(key in self.loss_weights for key in ["recon", "kl"]), \
            "Loss weights must include 'recon' and 'kl' keys."

    def forward(self, x):
        raise NotImplementedError

    def compute_loss(self, x, x_hat, z) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()
    

class DESOMModule(AutoencoderModule):  # TODO
    '''
    '''
    def __init__(
        self,
        net: nn.Module,
        som: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__(net, loss_fn, optimizer, lr_scheduler, loss_weights)
        self.save_hyperparameters(ignore=["net", "loss_fn"])
        self.som = som
        self.loss_weights = loss_weights or {"recon": 0.5, "distortion": 0.5}

        assert all(key in self.loss_weights for key in ["recon", "distortion"]), \
            "Loss weights must include 'recon' and 'distortion' keys."

    def forward(self, x):
        raise NotImplementedError

    def compute_loss(self, x, x_hat, z) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()
    
        # standard reconstruction loss
        recon_loss = self.loss_fn(x_hat, x)

        # distortion loss
        distortion_loss = None  # TODO: placeholder for distortion loss calculation

        # extendable loss dictionary
        losses = {"recon": recon_loss, "distortion": distortion_loss}

        # combine losses with weights
        total_loss = sum(self.loss_weights[k] * losses[k] for k in losses)
        losses["total"] = total_loss
        return losses