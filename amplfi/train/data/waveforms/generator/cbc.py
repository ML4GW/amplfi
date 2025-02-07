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
        """
        super().__init__(*args, **kwargs)
        self.right_pad = right_pad
        self.approximant = approximant
        self.waveform_generator = TimeDomainCBCWaveformGenerator(
            approximant,
            self.sample_rate,
            self.duration,
            f_min,
            f_ref,
            right_pad + self.fduration / 2,
        )

    def forward(self, **parameters) -> torch.Tensor:
        hc, hp = self.waveform_generator(**parameters)
        waveforms = torch.stack([hc, hp], dim=1)
        if self.time_translator is not None:
            waveforms = self.time_translator(waveforms)
        hc, hp = waveforms.transpose(1, 0)

        return hc.float(), hp.float()
