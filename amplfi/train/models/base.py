import logging
import sys
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from ml4gw.transforms import ChannelWiseScaler

from ..callbacks import ModelCheckpoint

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
        outdir: Path,
        learning_rate: float,
        weight_decay: float = 0.0,
        save_top_k_models: int = 10,
        patience: Optional[int] = None,
        checkpoint: Optional[Path] = None,
        verbose: bool = False,
    ):
        super().__init__()
        self._logger = self.init_logging(verbose)
        self.outdir = outdir
        outdir.mkdir(exist_ok=True, parents=True)
        self.inference_params = inference_params
        self.checkpoint = checkpoint
        self.save_hyperparameters()

        # initialize an unfit scaler here so that it is available
        # for the LightningModule to save and load from checkpoints
        self.scaler = ChannelWiseScaler(len(inference_params))

    def maybe_load_checkpoint(self, checkpoint: Optional[Path] = None):
        if checkpoint is not None:
            self._logger.info(
                f"Loading model weights from checkpoint path: {checkpoint}"
            )
            checkpoint = torch.load(checkpoint)
            self.load_state_dict(checkpoint["state_dict"])

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

        lr = self.hparams.learning_rate * world_size
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "valid_loss"},
        }

    def configure_callbacks(self):
        checkpoint = ModelCheckpoint(
            monitor="valid_loss",
            save_top_k=self.hparams.save_top_k_models,
            save_last=True,
            auto_insert_metric_name=False,
            mode="min",
        )
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        return [checkpoint, lr_monitor]
