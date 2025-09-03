from typing import TYPE_CHECKING, Optional

import torch

from ....prior import AmplfiPrior
from ..sampler import WaveformSampler

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

    def get_val_waveforms(
        self, _, world_size
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        num_waveforms = self.num_val_waveforms // world_size
        parameters = self.training_prior(num_waveforms, device="cpu")
        hc, hp = self(**parameters)
        return hc, hp, parameters

    def get_test_waveforms(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        parameters = self.testing_prior(self.num_test_waveforms)
        hc, hp = self(**parameters)
        return hc, hp, parameters

    def sample(
        self, X
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        N = len(X)
        parameters = self.training_prior(N, device=X.device)
        hc, hp = self(**parameters)
        return hc, hp, parameters

    def get_fit_parameters(self) -> torch.Tensor:
        parameters = self.training_prior(self.num_fit_params)
        return parameters

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
