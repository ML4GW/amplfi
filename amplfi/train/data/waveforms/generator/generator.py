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
        parameter_sampler: AmplfiPrior,
        test_parameter_sampler: Optional[AmplfiPrior] = None,
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
            test_parameter_sampler:
                A callable that takes an integer N and
                returns a dictionary of parameter Tensors, each of length `N`.
                Used for sampling test waveforms from a prior
                different from training data.
                If None, `parameter_sampler` is used.
            num_fit_params:
                The number of parameters used to fit standard scaler

        """
        super().__init__(*args, **kwargs)
        self.parameter_sampler = parameter_sampler
        self.test_parameter_sampler = (
            test_parameter_sampler or parameter_sampler
        )
        self.num_val_waveforms = num_val_waveforms
        self.num_test_waveforms = num_test_waveforms
        self.num_fit_params = num_fit_params

    def get_val_waveforms(self, _, world_size):
        num_waveforms = self.num_val_waveforms // world_size
        parameters = self.parameter_sampler(num_waveforms, device="cpu")
        hc, hp = self(**parameters)
        return hc, hp, parameters

    def get_test_waveforms(self):
        parameters = self.test_parameter_sampler(self.num_test_waveforms)
        hc, hp = self(**parameters)
        return hc, hp, parameters

    def sample(self, X):
        N = len(X)
        parameters = self.parameter_sampler(N, device=X.device)
        hc, hp = self(**parameters)
        return hc, hp, parameters

    def get_fit_params(self) -> torch.Tensor:
        parameters = self.parameter_sampler(self.num_fit_params)
        return parameters

    def forward(self):
        raise NotImplementedError
