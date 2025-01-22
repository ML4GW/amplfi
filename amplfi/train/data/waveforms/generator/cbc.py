from typing import Callable

import torch
from ml4gw.waveforms.generator import TimeDomainCBCWaveformGenerator

from .generator import WaveformGenerator


class CBCGenerator(WaveformGenerator):
    def __init__(
        self,
        *args,
        approximant: Callable,
        f_min: float,
        f_ref: float,
        right_pad: float,
        **kwargs,
    ):
        """
        A torch module for generating CBC waveforms on the fly.

        Args:
            waveform_generator:
                A TimeDomainCBCWaveformGenerator object
        """
        super().__init__(*args, **kwargs)
        self.waveform_generator = TimeDomainCBCWaveformGenerator(
            approximant,
            self.sample_rate,
            self.duration,
            f_min,
            f_ref,
            right_pad,
        )

    def forward(self, **parameters) -> torch.Tensor:
        hc, hp = self.waveform_generator(**parameters)
        return hc, hp
