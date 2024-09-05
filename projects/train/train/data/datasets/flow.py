import torch

from train.data.datasets.base import AmplfiDataset


class FlowDataset(AmplfiDataset):
    """
    Lightning DataModule for training normalizing flow networks
    """

    def inject(self, X, cross, plus, parameters):
        self.projector.to(self.device)
        self.whitener.to(self.device)

        X, psds = self.psd_estimator(X)
        waveforms = self.projector(
            parameters["dec"],
            parameters["psi"],
            parameters["phi"],
            cross=cross,
            plus=plus,
        )

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

        X += waveforms
        X = self.whitener(X, psds)

        # scale parameters
        parameters = self.scale(parameters)

        return X, parameters
