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
        A lightweight wrapper around
        `ml4gw.waveforms.generator.TimeDomainCBCWaveformGenerator`
        to make it compatible with
        `amplfi.train.data.waveforms.generator.WaveformGenerator`.


        Args:
            *args:
                Positional arguments passed to
                `amplfi.train.data.waveforms.generator.WaveformGenerator`
            approximant:
                A callable that takes parameter tensors
                and returns the cross and plus polarizations.
                For example, `ml4gw.waveforms.IMRPhenomD()`
            f_min:
                Lowest frequency at which waveform signal content
                is generated
            f_ref:
                Reference frequency
            right_pad:
                Position in seconds where coalesence is placed
                relative to the right edge of the window
            **kwargs:
                Keyword arguments passed to
                `amplfi.train.data.waveforms.generator.WaveformGenerator`
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
        # Define reference mass (adjust as needed)
        M0 = 1.0  # solar masses
        # Extract chirp mass and chirp distance
        chirp_mass = parameters.get("chirp_mass")
        chirp_distance = parameters.get("chirp_distance")
        if chirp_mass is not None and chirp_distance is not None:
            # Compute the physical distance
            distance = (chirp_mass / M0) ** (5/6) * chirp_distance
            # Update dictionary
            parameters["distance"] = distance
            del parameters["chirp_distance"]  # remove old key to avoid duplication
        # Generate waveform
        hc, hp = self.waveform_generator(**parameters)
        waveforms = torch.stack([hc, hp], dim=1)
        if self.time_translator is not None:
            waveforms = self.time_translator(waveforms)
        hc, hp = waveforms.transpose(1, 0)

        return hc.float(), hp.float()
