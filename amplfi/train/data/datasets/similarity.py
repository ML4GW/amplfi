from typing import Optional
import torch

from .base import AmplfiDataset, Tensor


class SimilarityDataset(AmplfiDataset):
    """
    Lightning DataModule that returns two views of the underlying signal
    decided by the augmentor. The augmentor is applied
    to the signal (waveform), and maybe be injected to the
    background. This is useful for
    self-supervised learning methods.

    Args:
        augmentor:
            A torch module that transforms waveforms according to some
            symmetry that is meant to be marginalized over.
        noiseless_view:
            If True, the augmentor is applied to the waveform first.
            For the reference view, this is then injected into background.
            The augmented view is just the augmented waveform alone.
            However, the augmented signal is still whitened with the same
            PSD as the reference view.
    """

    def __init__(
        self,
        *args,
        augmentor: torch.nn.Module,
        noiseless_view: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.noiseless_view = noiseless_view
        self.augmentor = augmentor

    def on_after_batch_transfer(
        self, batch, _
    ) -> tuple[Tensor, Tensor, dict[str, Tensor], Tensor]:
        if self.trainer.training:
            [batch] = batch
            cross, plus, parameters = self.waveform_sampler.sample(batch)
            [X_ref, X_aug], asds, parameters = self.inject(
                batch, cross, plus, parameters
            )

        elif self.trainer.validating or self.trainer.sanity_checking:
            [cross, plus, parameters], [background] = batch
            background = background[: len(cross)]
            keys = [
                k
                for k in self.hparams.inference_params
                if k not in ["dec", "psi", "phi"]
            ]
            parameters = {k: parameters[:, i] for i, k in enumerate(keys)}
            [X_ref, X_aug], asds, parameters = self.inject(
                background, cross, plus, parameters
            )

        elif self.trainer.testing:
            [cross, plus, parameters], [background] = batch
            keys = [
                k
                for k in self.hparams.inference_params
                if k not in ["dec", "psi", "phi"]
            ]
            parameters = {k: parameters[:, i] for i, k in enumerate(keys)}
            [X_ref, X_aug], asds, parameters = self.inject(
                background, cross, plus, parameters
            )
        return (X_ref, X_aug), asds, parameters

    def inject(self, X, cross, plus, parameters):
        X, psds = self.psd_estimator(X)
        dec, psi, phi = self.sample_extrinsic(X)

        waveforms = self.projector(dec, psi, phi, cross=cross, plus=plus)
        augmented = self.augmentor(waveforms)

        # append extrinsic parameters to parameters
        parameters.update({"dec": dec, "phi": phi, "psi": psi})

        # downselect to requested inference parameters
        parameters = {
            k: v
            for k, v in parameters.items()
            if k in self.hparams.inference_params
        }

        # make any requested parameter transforms
        parameters = self.transform(parameters)
        parameters = [
            torch.Tensor(parameters[k]) for k in self.hparams.inference_params
        ]
        parameters = torch.vstack(parameters).T
        if self.noiseless_view:
            X_ref = X + augmented
            X_aug = augmented
        else:
            X_ref = X + waveforms
            X_aug = X + augmented
        X_ref = self.whitener(X_ref, psds)
        X_aug = self.whitener(X_aug, psds)

        # scale parameters
        parameters = self.scale(parameters)

        # calculate asds and highpass
        freqs = torch.fft.rfftfreq(
            X_ref.shape[-1], d=1 / self.hparams.sample_rate
        )
        num_freqs = len(freqs)
        psds = torch.nn.functional.interpolate(
            psds, size=(num_freqs,), mode="linear"
        )

        mask = freqs > self.hparams.highpass
        psds = psds[:, :, mask]
        asds = torch.sqrt(psds)
        return [X_ref, X_aug], asds, parameters
