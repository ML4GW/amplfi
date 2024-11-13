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
        padding:
            Additional zero padding in seconds on top of `ringdown_duration`
            to add to the right of the waveform. So, the coalescence time
            of the waveform will be placed `ringdown_duration + padding`
            seconds from the right edge of the kernel. Defaults to 0.0.
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
        padding: float = 0.0,
        waveform_arguments: Dict[str, Any] = None,
        **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.waveform_arguments = waveform_arguments or {}

        # set approximant (possibly torch.nn.Module) as an attribute
        # so that it will get moved to the proper device when `.to` is called
        self.approximant = approximant
        self.waveform = partial(approximant, **waveform_arguments)
        self.f_min = f_min
        self.f_max = f_max
        self.padding = padding
        self.ringdown_duration = ringdown_duration

        frequencies = torch.linspace(0, self.nyquist, self.num_freqs)
        self.register_buffer("frequencies", frequencies)

    @property
    def right_pad_size(self):
        """
        Size of additional right padding in samples
        """
        return math.ceil(self.padding * self.sample_rate)

    @property
    def left_pad_size(self):
        """
        Size of left padding required to ensure
        the waveform is sufficiently long to slice
        according to the user requested `duration`
        """
        # calculate the size of the time domain
        # waveform after ffting
        freq_dim = self.freq_mask.sum()
        time_dim = 2 * (freq_dim - 1)
        # calculate the left padding required
        left_padding = self.num_samples - self.right_pad_size - time_dim
        return left_padding if left_padding > 0 else 0

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
        """

        device = parameters["chirp_mass"].device
        freqs = torch.clone(self.frequencies).to(device)
        self.approximant.to(device)

        parameters.update(self.waveform_arguments)

        hc, hp = self.approximant(freqs[self.freq_mask], **parameters)

        # fourier transform
        hc, hp = torch.fft.irfft(hc), torch.fft.irfft(hp)
        hc *= self.sample_rate
        hp *= self.sample_rate

        # roll the waveforms to join
        # the coalescence and ringdown
        ringdown_size = int(self.ringdown_duration * self.sample_rate)
        hc = torch.roll(hc, -ringdown_size)
        hp = torch.roll(hp, -ringdown_size)

        # pad the waveform on the right based on user specified padding;
        # pad the left side to ensure the waveform is long enough to slice
        hc = torch.nn.functional.pad(
            hc, (self.left_pad_size, self.right_pad_size, 0, 0)
        )
        hp = torch.nn.functional.pad(
            hp, (self.left_pad_size, self.right_pad_size, 0, 0)
        )

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
