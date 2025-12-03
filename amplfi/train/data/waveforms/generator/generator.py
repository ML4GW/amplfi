from typing import TYPE_CHECKING, Optional

import torch

from ....prior import AmplfiPrior
from ..sampler import WaveformSampler
from amplfi.train.data.utils.transforms import rescaled_distance_to_distance, chirp_distance_to_distance

if TYPE_CHECKING:
    pass


class WaveformGenerator(WaveformSampler):
    def __init__(
        self,
        *args,
        num_val_waveforms: int,
        num_test_waveforms: int,
        training_prior: AmplfiPrior,
        testing_prior: Optional[AmplfiPrior] = None,
        num_fit_params: int,
        M0: float = None,
        **kwargs,
    ):
        """
        A torch module for generating waveforms on the fly.

        Args:
            num_val_waveforms:
                Total number of validation waveforms to use.
                This total will be split up among all devices
            num_test_waveforms:
                Total number of testing waveforms to use.
                Testing is performed on one device.
            training_prior:
                A callable that takes an integer N and
                returns a dictionary of parameter Tensors, each of length `N`
            testing_prior:
                A callable that takes an integer N and
                returns a dictionary of parameter Tensors, each of length `N`.
                Used for sampling test waveforms from a prior
                different from training data.
                If None, `parameter_sampler` is used.
            num_fit_params:
                The number of parameters used to fit standard scaler

        """
        super().__init__(*args, **kwargs)
        self.training_prior = training_prior
        self.testing_prior = testing_prior or training_prior
        self.num_val_waveforms = num_val_waveforms
        self.num_test_waveforms = num_test_waveforms
        self.num_fit_params = num_fit_params
        self.M0 = M0

    def sample_extrinsic(self, X: torch.Tensor):
        """
        Sample extrinsic parameters used to project waveforms
        """
        N = len(X)
        dec = self.dec.sample((N,)).to(X.device)
        psi = self.psi.sample((N,)).to(X.device)
        phi = self.phi.sample((N,)).to(X.device)
        return dec, psi, phi

    def get_parameters(self, num, **kwargs):
        parameters = self.testing_prior(num, **kwargs)
        keys = list(parameters.keys())
        dec, psi, phi = self.sample_extrinsic(parameters[keys[0]])
        parameters["dec"] = dec
        parameters["psi"] = psi
        parameters["phi"] = phi

        if self.M0 is not None and "rescaled_distance" in keys:
            parameters["distance"] = rescaled_distance_to_distance(self.M0, **parameters)
            del parameters["rescaled_distance"]
        if self.M0 is not None and "chirp_distance" in keys:
            parameters["distance"] = chirp_distance_to_distance(self.M0, ifos=["H1", "L1", "V1"], **parameters)
            del parameters["chirp_distance"]
        return parameters

    def get_waveforms(
        self, num, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        parameters = get_parameters(num, **kwargs)
        hc, hp = self(**parameters)
        return hc, hp, parameters

    def get_val_waveforms(
        self, _, world_size
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        num_waveforms = self.num_val_waveforms // world_size
        return self.get_waveforms(num_test_waveforms, device="cpu")

    def get_test_waveforms(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        return self.get_waveforms(self.num_test_waveforms)     

    def sample(
        self, X
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        N = len(X)
        return self.get_waveforms(N, device=X.device)

    def get_fit_parameters(self) -> torch.Tensor:
        return get_parameters(self.num_fit_params)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
