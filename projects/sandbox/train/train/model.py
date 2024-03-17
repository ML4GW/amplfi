import logging
from pathlib import Path
from typing import Optional

import bilby
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from train.architectures.flows import FlowArchitecture
from train.callbacks import SaveAugmentedBatch
from train.testing import Result

Tensor = torch.Tensor


class PEModel(pl.LightningModule):
    """
    Args:
        arch:
            Neural network architecture to train.
            This should be a subclass of `FlowArchitecture`.
        patience:
            Number of epochs to wait for validation loss to improve
        learning_rate;
            Learning rate for the optimizer
        weight_decay:
            Weight decay for the optimizer
        save_top_k_models:
            Maximum number of best-performing model checkpoints
            to keep during training
    """

    def __init__(
        self,
        outdir: Path,
        arch: FlowArchitecture,
        learning_rate: float,
        weight_decay: float = 0.0,
        num_samples_draw: int = 1000,
        num_corner: int = 10,
        patience: Optional[int] = None,
        save_top_k_models: int = 10,
    ) -> None:
        super().__init__()
        # construct our model up front and record all
        # our hyperparameters to our logdir
        self.model = arch
        outdir.mkdir(exist_ok=True, parents=True)
        self.outdir = outdir
        self.num_samples_draw = num_samples_draw
        self.num_corner = num_corner
        self.save_hyperparameters(ignore=["arch"])
        self._logger = self.get_logger()

    def get_logger(self):
        logger_name = "PEModel"
        return logging.getLogger(logger_name)

    def on_fit_start(self):
        datamodule = self.trainer.datamodule
        for item in datamodule.__dict__.values():
            if isinstance(item, torch.nn.Module):
                item.to(self.device)

    def on_test_start(self):
        datamodule = self.trainer.datamodule
        for item in datamodule.__dict__.values():
            if isinstance(item, torch.nn.Module):
                item.to(self.device)

    def forward(self, parameters, strain) -> Tensor:
        return -self.model.log_prob(parameters, context=strain)

    def training_step(self, batch, _):
        strain, parameters = batch
        loss = self(parameters, strain).mean()
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
            logger=True,
        )
        return loss

    def validation_step(self, batch, _):
        strain, parameters = batch
        loss = self(parameters, strain).mean()
        self.log(
            "valid_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        return loss

    def cast_as_bilby_result(
        self,
        samples: np.ndarray,
        truth: np.ndarray,
    ):
        """Cast posterior samples as Bilby Result object
        for ease of producing corner and pp plots

        Args:
            samples: posterior samples (1, num_samples, num_params)
            truth: true values of the parameters  (1, num_params)
            priors: dictionary of prior objects
            label: label for the bilby result object

        """

        inference_params = self.trainer.datamodule.inference_params
        injection_parameters = {
            k: float(v) for k, v in zip(inference_params, truth)
        }

        # create dummy prior with correct attributes
        # for making our results compatible with bilbys make_pp_plot
        priors = {
            param: bilby.core.prior.base.Prior(latex_label=param)
            for param in inference_params
        }
        posterior = dict()
        for idx, k in enumerate(inference_params):
            posterior[k] = samples.T[idx].flatten()
        posterior = pd.DataFrame(posterior)

        return Result(
            label="PEModel",
            injection_parameters=injection_parameters,
            posterior=posterior,
            search_parameter_keys=inference_params,
            priors=priors,
        )

    def on_test_epoch_start(self):
        self.test_results = []
        self.num_plotted = 0

    def test_step(self, batch, batch_idx):
        # whitened strain and de-scaled parameters
        strain, parameters = batch
        samples = self.model.sample(
            self.hparams.num_samples_draw, context=strain
        )
        descaled = self.trainer.datamodule.scale(samples, reverse=True)
        result = self.cast_as_bilby_result(
            descaled.cpu().numpy(),
            parameters.cpu().numpy()[0],
        )
        self.test_results.append(result)

        if batch_idx % 100 == 0 and self.num_plotted < self.num_corner:
            skymap_filename = self.outdir / f"{self.num_plotted}_mollview.png"
            corner_filename = self.outdir / f"{self.num_plotted}_corner.png"
            result.plot_corner(
                save=True,
                filename=corner_filename,
                levels=(0.5, 0.9),
            )
            result.plot_mollview(
                outpath=skymap_filename,
            )
            self.num_plotted += 1

    def on_test_epoch_end(self):
        bilby.result.make_pp_plot(
            self.test_results,
            save=True,
            filename=self.outdir / "pp-plot.png",
            keys=self.trainer.datamodule.inference_params,
        )
        del self.test_results, self.num_plotted

    def configure_callbacks(self):
        checkpoint = ModelCheckpoint(
            monitor="valid_loss",
            save_top_k=self.hparams.save_top_k_models,
            save_last=True,
            auto_insert_metric_name=False,
            mode="min",
        )
        return [SaveAugmentedBatch(), checkpoint]

    def configure_optimizers(self):
        if not torch.distributed.is_initialized():
            world_size = 1
        else:
            world_size = torch.distributed.get_world_size()

        lr = self.hparams.learning_rate * world_size
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "valid_loss"},
        }
