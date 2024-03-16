import logging
from dataclasses import dataclass
from typing import Optional

import bilby
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from train.architectures.flows import FlowArchitecture
from train.callbacks import SaveAugmentedBatch

Tensor = torch.Tensor


@dataclass
class PriorKey:
    latex_label: str


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
        arch: FlowArchitecture,
        learning_rate: float,
        weight_decay: float = 0.0,
        num_samples_draw: int = 1000,
        patience: Optional[int] = None,
        save_top_k_models: int = 10,
    ) -> None:
        super().__init__()
        # construct our model up front and record all
        # our hyperparameters to our logdir
        self.model = arch
        self.save_hyperparameters(ignore=["arch"])
        self._logger = self.get_logger()

    def get_logger(self):
        logger_name = "PEModel"
        return logging.getLogger(logger_name)

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

        # create dummy prior for bilby result
        dummy_prior = {
            param: PriorKey(latex_label=param) for param in inference_params
        }
        posterior = dict()
        for idx, k in enumerate(inference_params):
            posterior[k] = samples.T[idx].flatten()
        posterior = pd.DataFrame(posterior)

        return bilby.result.Result(
            label="PEModel",
            injection_parameters=injection_parameters,
            posterior=posterior,
            search_parameter_keys=inference_params,
            priors=dummy_prior,
        )

    def on_test_epoch_start(self):
        self.test_results = []

    def test_step(self, batch, batch_idx):
        # whitened strain and de-scaled parameters
        strain, parameters = batch

        samples = self.model.sample(
            [1, self.hparams.num_samples_draw], context=strain
        )
        descaled = self.trainer.datamodule.scale(samples[0], reverse=True)

        result = self.cast_as_bilby_result(
            descaled,
            parameters,
        )
        self.test_results.append(result)

        # TODO corner plots

    def on_test_epoch_end(self):
        bilby.result.make_pp_plot(
            self.test_results,
            save=True,
            filename="pp-plot.png",
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
