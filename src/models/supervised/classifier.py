import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torchmetrics

from typing import Dict, Optional, Any

class Classifier(L.LightningModule):
    def __init__(
            self, 
            backbone: nn.Module,
            num_classes: int, 
            optimizer: torch.optim.Optimizer,
            target_label: str = "instrument_family", 
            lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            lr_scheduler_params: Optional[Dict[str, Any]] = None,
        ):
        """
        Generic supervised classifier.

        Args:
            backbone (nn.Module): Encoder network.
            target_label (str): Name of the label to classify.
            num_classes (int): Number of output classes.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
            lr_scheduler_params (dict, optional): Parameters for the learning rate scheduler.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])

        self.backbone = backbone
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.target_label = target_label
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params or {}

        self.classifier = None
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        self.embeddings = []
        self.labels = []

    def forward(self, x):
        z = self.backbone(x)

        # initialize classifier lazily if needed
        if self.classifier is None:
            output_dim = z.shape[-1]
            self.classifier = nn.Linear(output_dim, self.num_classes).to(z.device)

        logits = self.classifier(z)
        return logits, z
    
    def compute_loss(self, logits, targets):
        return self.loss_fn(logits, targets)
    
    def model_step(self, batch, batch_idx):
        x = batch["features"]
        labels = batch["labels"]
        targets = labels[self.target_label]
        logits, z = self(x)
        loss = self.compute_loss(logits, targets)
        return x, logits, z, targets, labels, loss

    def training_step(self, batch, batch_idx):
        x, logits, z, targets, labels, loss = self.model_step(batch, batch_idx)
        self.log(f"train/loss/{self.target_label}", loss, prog_bar=True)
        return {"loss": loss, "z": z, "labels": labels}

    def validation_step(self, batch, batch_idx):
        x, logits, z, targets, labels, loss = self.model_step(batch, batch_idx)
        acc = self.accuracy(logits, targets)
        self.log(f"val/loss/{self.target_label}", loss, prog_bar=True)
        self.log(f"val/acc/{self.target_label}", acc, prog_bar=True)
        return {"loss": loss, "z": z, "labels": labels}

    def test_step(self, batch, batch_idx):
        x, logits, z, targets, labels, loss = self.model_step(batch, batch_idx)
        acc = self.accuracy(logits, targets)
        self.log(f"test/loss/{self.target_label}", loss, prog_bar=True)
        self.log(f"test/acc/{self.target_label}", acc, prog_bar=True)
        return {"loss": loss, "z": z, "labels": labels}

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        self.embeddings.append(outputs["z"])
        self.labels.append(outputs["labels"])
    
    def on_validation_epoch_end(self):
        aggregated_labels = {}  # gather labels from across batches
        for d in self.labels:
            for key, arr in d.items():
                if key not in aggregated_labels:
                    aggregated_labels[key] = []
                aggregated_labels[key].append(arr)

        self.stacked_labels = {key: torch.hstack(arr).detach()
                    for key, arr in aggregated_labels.items()}

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
