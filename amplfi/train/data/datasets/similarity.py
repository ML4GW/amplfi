import torch

from .base import AmplfiDataset


class SimilarityDataset(AmplfiDataset):
    """
    Lightning DataModule for training similarity networks

    Args:
        augmentor:
            A torch module that transforms waveforms according to some
            symmetry that is meant to be marginalized over.
    """

    def __init__(self, *args, augmentor: torch.nn.Module, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentor = augmentor

    def inject(self, X, cross, plus, parameters):

        X, psds = self.psd_estimator(X)
        dec, psi, phi = self.waveform_sampler.sample_extrinsic(X)

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
