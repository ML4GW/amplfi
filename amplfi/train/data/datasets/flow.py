import torch

from .base import AmplfiDataset
from ml4gw import gw

Tensor = torch.Tensor


class FlowDataset(AmplfiDataset):
    """
    Lightning DataModule for training normalizing flow networks
    """

    def inject(
        self,
        X: Tensor,
        cross: Tensor,
        plus: Tensor,
        parameters: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        self.projector.to(self.device)
        self.whitener.to(self.device)

        X, psds = self.psd_estimator(X)
        dec, psi, phi = self.sample_extrinsic(X)
        waveforms = self.projector(dec, psi, phi, cross=cross, plus=plus)

        # append extrinsic parameters to parameters
        parameters.update({"dec": dec, "psi": psi, "phi": phi})

        if self.trainer.testing:
            # save randomly sampled extrinisic parameters
            # used for saving to disk;
            # TODO: this is a bit hacky, but would require some
            # refactoring to address
            for key in ["dec", "psi", "phi"]:
                self.test_extrinsic[key].extend(parameters[key].tolist())

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

        num_freqs = waveforms.shape[-1] // 2 + 1
        psds = torch.nn.functional.interpolate(
            psds, size=(num_freqs,), mode="linear"
        )
        snrs = gw.compute_network_snr(
            waveforms,
            psds,
            self.hparams.sample_rate,
            self.hparams.highpass,
        )

        X += waveforms
        X = self.whitener(X, psds)

        # scale parameters
        parameters = self.scale(parameters)

        # calculate asds and highpass
        freqs = torch.fft.rfftfreq(X.shape[-1], d=1 / self.hparams.sample_rate)
        num_freqs = len(freqs)
        psds = torch.nn.functional.interpolate(
            psds, size=(num_freqs,), mode="linear"
        )

        mask = freqs > self.hparams.highpass
        psds = psds[:, :, mask]
        asds = torch.sqrt(psds)

        return X, asds, parameters, snrs
