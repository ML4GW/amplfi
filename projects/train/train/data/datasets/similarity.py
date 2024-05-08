import torch
from train.data.datasets.base import AmplfiDataset


class SimilarityDataset(AmplfiDataset):
    def __init__(self, *args, augmentor: torch.nn.Module, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentor = augmentor

    def inject(self, X, cross, plus, parameters):
        N = len(X)
        X, psds = self.psd_estimator(X)
        dec, psi, phi = self.waveform_sampler.sample_extrinsic(
            N, device=X.device
        )
        waveforms = self.projector(dec, psi, phi, cross=cross, plus=plus)
        augmented = self.augmentor(waveforms)
        waveforms = self.waveform_sampler.slice_waveforms(waveforms)
        augmented = self.waveform_sampler.slice_waveforms(augmented)

        # append extrinisc parameters to parameters
        parameters.update({"dec": dec, "phi": phi, "psi": psi})

        # downselect to requested inference parameters
        parameters = {
            k: v for k, v in parameters.items() if k in self.inference_params
        }

        # make any requested parameter transforms
        parameters = self.transform(parameters)
        parameters = [torch.Tensor(v) for v in parameters.values()]
        parameters = torch.vstack(parameters).T

        X_ref = X + waveforms
        X_aug = X + augmented
        X_ref = self.whitener(X_ref, psds)
        X_aug = self.whitener(X_aug, psds)

        # scale parameters
        parameters = self.scale(parameters)

        return X_ref, X_aug, parameters
