import torch
from train.losses import VICRegLoss
from train.models.base import AmplfiModel


class SimilarityModel(AmplfiModel):
    """
    A LightningModule for training similarity embeddings

    Args:
        embedding:
            A neural network architecture that maps waveforms
            to lower dimensional embedded space
        augmentor:
            A torch.nn.Module that augments waveforms
            for training similarity embeddings. The embedding
            will be trained to be invariant to the augmentation
    """

    def __init__(
        self,
        *args,
        embedding: torch.nn.Module,
        augmentor: torch.nn.Module,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # TODO: parmeterize cov, std, repr weights
        self.embedding = embedding
        self.loss = VICRegLoss()

    def forward(self, ref, aug):
        ref = self.embedding(ref)
        aug = self.embedding(aug)
        loss, *_ = self.vicreg_loss(ref, aug)
        return loss

    def validation_step(self, batch, _):
        ref, aug = batch
        loss = self(ref, aug)
        self.log(
            "valid_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True
        )

    def training_step(self, batch, _):
        ref, aug = batch
        loss = self(ref, aug)
        self.log(
            "train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=False
        )
        return loss
