from lightning.pytorch.callbacks import Callback
import viz

from timbremetrics import TimbreMetric
import numpy as np
import matplotlib.pyplot as plt

from lightning.pytorch.utilities import rank_zero_only
import torch.distributed as distr

from lightning.pytorch.loggers import WandbLogger
import wandb

class TimbreMetrics(Callback):
    def __init__(self, 
                 sample_rate: int,
                 fixed_duration: float,
                 device: str,
                 heatmap_kw: dict = {},
                 ):
        super().__init__()
        self.metric = TimbreMetric(sample_rate=sample_rate,
                                   fixed_duration=fixed_duration,
                                   device=device,)
        
        self.imshow_kw = heatmap_kw.get('imshow_kw', {})
        self.text_kw = heatmap_kw.get('text_kw', {})

    @rank_zero_only
    def visualize(self, trainer, results):
        distances = []
        metrics = []
        
        if self.metric.device == "cuda":
            for dist, metric in results.items():
                distances.append(dist)
                for metric, value in metric.items():
                        metrics.append(metric) if metric not in metrics else None
                        results[dist][metric] = value.cpu().numpy()

        similarities = np.vstack([list(metric.values()) for metric in results.values()])

        fig, ax = plt.subplots()
        im, cbar = viz.heatmap(similarities, distances, metrics, ax=ax,
                        cmap="YlGn", cbarlabel="similarity", **self.imshow_kw)
        texts = viz.annotate_heatmap(im, valfmt="{x:.3f}", **self.text_kw)
        fig.tight_layout()

        trainer.logger.experiment.log(
            {f"timbremetrics/viz": wandb.Image(fig)  # plotly begone
                if isinstance(trainer.logger, WandbLogger) else fig,
             "epoch": trainer.current_epoch,
            },
        )

        plt.close(fig)

    def on_validation_end(self, trainer, pl_module):
        local_results = self.metric(pl_module)

        # gather results from all processes
        if distr.is_initialized() and distr.get_world_size() > 1:
            world_size = distr.get_world_size()
            gathered_results = [None for _ in range(world_size)]
            distr.gather_object(local_results, gathered_results)

            # TODO: not yet tested with multiple processes!!
            local_results = {}
            for results in gathered_results:
                for dist, metrics in results.items():
                    if dist not in local_results:
                        local_results[dist] = {}
                    for metric, value in metrics.items():
                        if metric not in local_results[dist]:
                            local_results[dist][metric] = value

        self.visualize(trainer, local_results)