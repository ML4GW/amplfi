from ..architectures.similarity import SimilarityEmbedding
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
        arch: SimilarityEmbedding,
        similarity_loss: VICRegLoss,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # TODO: parmeterize cov, std, repr weights
        self.model = arch
        self.similarity_loss = similarity_loss

        # if checkpoint is not None, load in model weights;
        # checkpoint should only be specified in this way
        # if running trainer.test
        self.maybe_load_checkpoint(self.checkpoint)

    def forward(
        self,
        ref,
        aug,
    ):
        ref = self.model(ref)
        aug = self.model(aug)
        loss, (inv_loss, var_loss, cov_loss) = self.similarity_loss(ref, aug)
        return loss, (inv_loss, var_loss, cov_loss)

    def validation_step(self, batch, _):
        [ref, aug], asds, _ = batch
        loss, (inv_loss, var_loss, cov_loss) = self((ref, asds), (aug, asds))
        self.log(
            "valid_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True
        )

        self.log("valid_var_loss", var_loss, on_step=False, on_epoch=True)
        self.log("valid_cov_loss", cov_loss, on_step=False, on_epoch=True)
        self.log("valid_inv_loss", inv_loss, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, _):
        # unpack batch - can ignore parameters
        [ref, aug], asds, _ = batch

        # pass reference and augmented data contexts
        # through embedding and calculate similarity loss
        loss, (inv_loss, var_loss, cov_loss) = self((ref, asds), (aug, asds))

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log("train_var_loss", var_loss, on_step=True, on_epoch=True)
        self.log("train_cov_loss", cov_loss, on_step=True, on_epoch=True)
        self.log("train_inv_loss", inv_loss, on_step=True, on_epoch=True)

        return loss

    def configure_callbacks(self):
        callbacks = super().configure_callbacks()
        callbacks.append(SaveAugmentedSimilarityBatch())
        return callbacks
