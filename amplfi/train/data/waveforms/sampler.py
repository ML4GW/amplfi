from typing import Optional

import torch

from ..augmentors import TimeTranslator
from ..utils.utils import ParameterTransformer

Distribution = torch.distributions.Distribution


class WaveformSampler(torch.nn.Module):
    """
    Base object defining methods that waveform producing classes
    should implement. Should not be instantiated on its own.

    Args:
        fduration:
            Desired length in seconds of the time domain
            response of the whitening filter built from PSDs.
            See `ml4gw.spectral.truncate_inverse_power_spectrum`
        kernel_length:
            Length in seconds of window passed to neural network.
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
        parameter_transformer:
            A `ParameterTransformer` object that applies any
            additional transformations to parameters before
            they are scaled and passed to the neural network.

    """

    def __init__(
        self,
        *args,
        fduration: float,
        kernel_length: float,
        sample_rate: float,
        inference_params: list[str],
        dec: Distribution,
        psi: Distribution,
        phi: Distribution,
        jitter: Optional[float] = None,
        parameter_transformer: Optional[ParameterTransformer] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.parameter_transformer = parameter_transformer or (lambda x: x)
        self.inference_params = inference_params
        self.fduration = fduration
        self.kernel_length = kernel_length

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
    def duration(self):
        """
        Length of kernel before whitening removes
        fduration / 2 from each side
        """
        return self.fduration + self.kernel_length

    def get_val_waveforms(self):
        raise NotImplementedError

    def get_test_waveforms(self):
        raise NotImplementedError

    def fit_scaler(self, scaler: torch.nn.Module) -> torch.nn.Module:
        raise NotImplementedError

    def sample(self, X: torch.Tensor):
        """Defines how to sample waveforms for training"""
        raise NotImplementedError
