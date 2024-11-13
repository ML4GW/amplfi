from typing import Optional

import torch

from ..augmentors import TimeTranslator
from ..utils.utils import ParameterTransformer

Distribution = torch.distributions.Distribution


class WaveformSampler(torch.nn.Module):
    """
    Base object for producing waveforms for training validating and testing

    Args:
        duration:
            The length of the waveform in seconds to return. This
            includes kernel that the network analyzes, and extra data
            lost due to whitening filter settle in.
        sample_rate:
            Sample rate in Hz of generated waveforms
        inference_params:
            The parameters the model will perform inference on
        dec:
            The distribution of declinations to sample from
        psi:
            The distribution of polarization angles to sample from
        phi:
            The distribution of "right ascensions" to sample from
        jitter:
            The amount of jitter in seconds to randomly shift
            the waveform coalescence time. If `None`, no jitter is applied.
    """

    def __init__(
        self,
        duration: float,
        sample_rate: float,
        inference_params: list[str],
        dec: Distribution,
        psi: Distribution,
        phi: Distribution,
        jitter: Optional[float] = None,
        parameter_transformer: Optional[ParameterTransformer] = None,
    ) -> None:

        super().__init__()
        self.parameter_transformer = parameter_transformer or (lambda x: x)
        self.inference_params = inference_params
        self.duration = duration
        self.sample_rate = sample_rate
        self.dec, self.psi, self.phi = dec, psi, phi
        self.time_translator = (
            TimeTranslator(jitter, sample_rate) if jitter is not None else None
        )

    def sample_extrinsic(self, X: torch.Tensor):
        """
        Sample extrinsic parameters used to project waveforms
        """
        N = len(X)
        dec = self.dec.sample((N,)).to(X.device)
        psi = self.psi.sample((N,)).to(X.device)
        phi = self.phi.sample((N,)).to(X.device)
        return dec, psi, phi

    @property
    def waveform_size(self):
        return int(self.duration * self.sample_rate)

    def get_val_waveforms(self):
        raise NotImplementedError

    def get_test_waveforms(self):
        raise NotImplementedError

    def fit_scaler(self, scaler: torch.nn.Module) -> torch.nn.Module:
        raise NotImplementedError

    def sample(self, X: torch.Tensor):
        """Defines how to sample waveforms for training"""
        raise NotImplementedError
