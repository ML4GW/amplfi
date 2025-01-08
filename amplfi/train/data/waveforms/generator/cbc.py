import math
from functools import partial
from typing import Any, Callable, Dict

import torch

from .generator import WaveformGenerator


class FrequencyDomainCBCGenerator(WaveformGenerator):
    """
    A torch module for generating waveforms on the fly.

    Args:
        waveform:
            A callable that returns cross and plus polarizations
            given a set of parameters.
        duration:
            The duration of the waveform in seconds
        sample_rate:
            Sample rate of the waveform in Hz
        f_min:
            The minimum frequency of the waveform in Hz
        f_max:
            The maximum frequency of the waveform in Hz
        ringdown_duration:
            The duration of time in seconds to roll the fft'd
            waveform to the left to join the coalescence and ringdown.
            This will place the coalescence time `ringdown_duration` seconds
            from the right edge of the waveform. Defaults to 0.5.
        waveform_arguments:
            A dictionary of fixed arguments to pass to the waveform model,
            e.g. `f_ref` for CBC waveforms
    """

    def __init__(
        self,
        *args,
        approximant: Callable,
        f_min: float = 0.0,
        f_max: float = 0.0,
        ringdown_duration: float = 0.5,
        waveform_arguments: Dict[str, Any] = None,
        **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.waveform_arguments = waveform_arguments or {}

        # set approximant (possibly torch.nn.Module) as an attribute
        # so that it will get moved to the proper device when `.to` is called
        self.approximant = approximant
        self.waveform = partial(approximant, **self.waveform_arguments)
        self.f_min = f_min
        self.f_max = f_max
        self.ringdown_duration = ringdown_duration

        frequencies = torch.linspace(0, self.nyquist, self.num_freqs)
        self.register_buffer("frequencies", frequencies)

    @property
    def nyquist(self):
        return self.sample_rate / 2

    @property
    def num_samples(self):
        # number of samples in the time domain
        return math.ceil(self.duration * self.sample_rate)

    @property
    def num_freqs(self):
        # number of frequencies bins
        return self.num_samples // 2 + 1

    @property
    def freq_mask(self):
        return (self.frequencies >= self.f_min) * (
            self.frequencies < self.f_max
        )

    def time_domain_strain(self, **parameters):
        """
        Generate time domain strain from a given set of parameters.
        If waveform is in the frequency domain,
        it will be transformed via an inverse fourier transform.

        Args:
            parameters:
                A dictionary of parameters to pass to the waveform model
        """

        device = parameters["chirp_mass"].device
        freqs = torch.clone(self.frequencies).to(device)
        self.approximant.to(device)

        parameters.update(self.waveform_arguments)

        # generate hc and hp at specified frequencies
        hc, hp = self.approximant(freqs[self.freq_mask], **parameters)

        # create spectrum of frequencies, initially filled with zeros,
        # with a delta_f such that after we fft to time domain the duration
        # of the waveform will be `self.duration`
        shape = (hc.shape[0], self.num_freqs)
        hc_spectrum = torch.zeros(shape, dtype=hc.dtype, device=device)
        hp_spectrum = torch.zeros(shape, dtype=hc.dtype, device=device)

        # fill the spectrum with the
        # hc and hp values at the specified frequencies
        hc_spectrum[:, self.freq_mask] = hc
        hp_spectrum[:, self.freq_mask] = hp

        # now, irfft and scale the waveforms by sample_rate
        hc, hp = torch.fft.irfft(hc_spectrum), torch.fft.irfft(hp_spectrum)
        hc *= self.sample_rate
        hp *= self.sample_rate

        # roll the waveforms to join the coalescence and ringdown;
        # account for the data lost due to the whitening filter;
        # after whitening, the coalescence time will be placed
        # `self.ringdown_duration` seconds from the right edge
        roll_size = int(
            (self.ringdown_duration + self.fduration / 2) * self.sample_rate
        )

        hc = torch.roll(hc, -roll_size, dims=-1)
        hp = torch.roll(hp, -roll_size, dims=-1)

        return hc, hp

    def frequency_domain_strain(self, **parameters):
        device = parameters["chirp_mass"].device
        freqs = torch.clone(self.frequencies).to(device)
        self.approximant.to(device)
        return self.waveform(freqs[self.freq_mask], **parameters)

    def slice_waveforms(self, waveforms: torch.Tensor):
        # for cbc waveforms, the padding (see above)
        # determines where the coalescence time lies
        # relative to the right edge, so just subtract
        # the pre-whiten kernel size from the right edge and slice
        start = waveforms.shape[-1] - self.waveform_size
        waveforms = waveforms[..., start:]
        if self.time_translator is not None:
            waveforms = self.time_translator(waveforms)
        return waveforms

    def forward(self, **parameters):
        hc, hp = self.time_domain_strain(**parameters)
        waveforms = torch.stack([hc, hp], dim=1)
        waveforms = self.slice_waveforms(waveforms)
        hc, hp = waveforms.transpose(1, 0)
        return hc, hp
