import logging
import math
import sys
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
import torch
from ml4gw.transforms import ChannelWiseScaler

Tensor = torch.Tensor
Distribution = torch.distributions.Distribution


class AmplfiModel(pl.LightningModule):
    """
    Amplfi model base class

    Encodes common functionality for all models,
    such as on-device augmentation and preprocessing,

    Args:
        checkpoint:
            Path to a model checkpoint to load. This will load in weights
            for both flow and embedding. Should only be specified when
            running `trainer.test`. For resuming a `trainer.fit` run from
            a checkpoint, use the --ckpt_path `Trainer` argument.
    """

    def __init__(
        self,
        inference_params: list[str],
        learning_rate: float,
        train_outdir: Path,
        test_outdir: Optional[Path] = None,
        weight_decay: float = 0.0,
        patience: int = 10,
        factor: float = 0.1,
        verbose: bool = False,
    ):
        super().__init__()
        self.scheduler_patience = patience
        self.scheduler_factor = factor
        self._logger = self.init_logging(verbose)

        if test_outdir is None:
            test_outdir = train_outdir / "test_results"

        train_outdir.mkdir(exist_ok=True)
        test_outdir.mkdir(exist_ok=True)
        self.test_outdir = test_outdir
        self.train_outdir = train_outdir

        self.inference_params = inference_params

        self.save_hyperparameters()

        # initialize an unfit scaler here so that it is available
        # for the LightningModule to save and load from checkpoints
        self.scaler = ChannelWiseScaler(len(inference_params))

    def init_logging(self, verbose):
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            format=log_format,
            level=logging.DEBUG if verbose else logging.INFO,
            stream=sys.stdout,
        )

        world_size, rank = self.get_world_size_and_rank()
        logger_name = self.__class__.__name__
        if world_size > 1:
            logger_name += f":{rank}"
        return logging.getLogger(logger_name)

    def get_world_size_and_rank(self) -> tuple[int, int]:
        """
        Name says it all, but generalizes to the case
        where we aren't running distributed training.
        """
        if not torch.distributed.is_initialized():
            return 1, 0
        else:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            return world_size, rank

    def setup(self, stage):
        if stage == "fit":
            # if we're fitting, store an instance of the
            # fit scaler in the model
            # so that its weights can be checkpointed
            self.scaler = self.trainer.datamodule.scaler

    def configure_optimizers(self):
        if not torch.distributed.is_initialized():
            world_size = 1
        else:
            world_size = torch.distributed.get_world_size()

        lr = self.hparams.learning_rate * math.sqrt(world_size)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.scheduler_patience,
            factor=self.scheduler_factor,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "valid_loss"},
        }

    def scale(self, parameters, reverse: bool = False):
        """
        Apply standard scaling to transformed parameters
        """
        parameters = parameters.transpose(1, 0)
        scaled = self.scaler(parameters, reverse=reverse)
        scaled = scaled.transpose(1, 0)
        return scaled
