import math
from functools import partial
from typing import Any, Callable, Dict

import torch

from train.data.waveforms.generator.generator import WaveformGenerator


class FrequencyDomainCBCGenerator(WaveformGenerator):
    """
    A torch module for generating CBC waveforms on the fly using
    ml4gw waveform models.

    Args:
    """

    def __init__(
        self,
        *args,
        approximant: Callable,
        f_min: float = 0.0,
        f_max: float = 0.0,
        padding: float = 0.0,
        waveform_arguments: Dict[str, Any] = None,
        **kwargs
    ):
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
            padding:
                The amount of padding to add to the right
                of the waveform in seconds
            waveform_arguments:
                A dictionary of fixed arguments to pass to the waveform model,
                e.g. `f_ref` for CBC waveforms
        """
        super().__init__(*args, **kwargs)
        waveform_arguments = waveform_arguments or {}
        self.waveform = partial(approximant, **waveform_arguments)
        self.f_min = f_min
        self.f_max = f_max
        self.padding = padding

        frequencies = torch.linspace(0, self.nyquist, self.num_freqs)
        self.register_buffer("frequencies", frequencies)

    @property
    def pad_size(self):
        return int(self.padding * self.sample_rate)

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

    @property
    def times(self):
        pass

    def time_domain_strain(self, **parameters):
        """
        Generate time domain strain from a given set of parameters.
        If waveform is in the frequency domain,
        it will be transformed via an inverse fourier transform.

        Args:
            parameters:
                A dictionary of parameters to pass to the waveform model
            padding:
                The amount of padding to add to
                the right of the waveform in seconds
        """

        # TODO: support time domain waveforms
        hc, hp = self.waveform(self.frequencies[self.freq_mask], **parameters)

        # fourier transform
        hc, hp = torch.fft.irfft(hc), torch.fft.irfft(hp)
        hc *= self.sample_rate
        hp *= self.sample_rate

        # TODO: any windowing? support left padding?

        hc = torch.nn.functional.pad(hc, (0, self.pad_size, 0, 0))
        hp = torch.nn.functional.pad(hp, (0, self.pad_size, 0, 0))

        return hc, hp

    def frequency_domain_strain(self, **parameters):
        return self.waveform(self.frequencies[self.freq_mask], **parameters)

    def slice_waveforms(self, waveforms: torch.Tensor):
        # for cbc waveforms, the padding (see above)
        # determines where the coalescence time lies
        # relative to the right edge, so just subtract
        # the pre-whiten kernel size from the right edge and slice
        start = waveforms.shape[-1] - self.waveform_size
        return waveforms[..., start:]

    def forward(self, **parameters):
        hc, hp = self.time_domain_strain(**parameters)
        waveforms = torch.stack([hc, hp], dim=1)
        waveforms = self.slice_waveforms(waveforms)
        hc, hp = waveforms.transpose(1, 0)
        return hc, hp
