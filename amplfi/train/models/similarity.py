import torch

from ..callbacks import SaveAugmentedSimilarityBatch
from ..losses import VICRegLoss
from .base import AmplfiModel


class SimilarityModel(AmplfiModel):
    """
    A LightningModule for training similarity embeddings

    Args:
        arch:
            A neural network architecture that maps waveforms
            to lower dimensional embedded space
    """

    def __init__(
        self,
        *args,
        arch: torch.nn.Module,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # TODO: parmeterize cov, std, repr weights
        self.model = arch
        self.loss = VICRegLoss()

        # if checkpoint is not None, load in model weights;
        # checkpoint should only be specified in this way
        # if running trainer.test
        self.maybe_load_checkpoint(self.checkpoint)

    def forward(self, ref, aug):
        ref = self.model(ref)
        aug = self.model(aug)
        loss, *_ = self.loss(ref, aug)
        return loss

    def validation_step(self, batch, _):
        [ref, aug], _ = batch
        loss = self(ref, aug)
        self.log(
            "valid_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return loss

    def training_step(self, batch, _):
        [ref, aug], _ = batch
        loss = self(ref, aug)
        self.log(
            "train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=False
        )
        return loss

    def configure_callbacks(self):
        callbacks = super().configure_callbacks()
        callbacks.append(SaveAugmentedSimilarityBatch())
        return callbacks
