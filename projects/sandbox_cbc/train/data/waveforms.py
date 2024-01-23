from math import ceil

import torch

from ml4gw import waveforms


class FrequencyDomainWaveformGenerator:
    """Loose implementation of bilby's WaveformGenerator using torch"""

    def __init__(
        self,
        time_duration: float,
        sampling_frequency: float,
        f_min: float,
        f_max: float,
        approximant: waveforms.IMRPhenomD,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.sampling_frequency = sampling_frequency
        number_of_samples = ceil(time_duration * sampling_frequency)
        number_of_frequencies = number_of_samples // 2 + 1

        self.frequency_array = torch.linspace(
            0,
            sampling_frequency / 2,
            number_of_frequencies,
            device=self.device,
        )
        self.time_array = torch.linspace(
            0,
            time_duration - 1 / sampling_frequency,
            number_of_samples,
            device=self.device,
        )
        self.f_min = f_min
        self.f_max = f_max

        if isinstance(approximant, str):
            try:
                self.approximant = getattr(waveforms, approximant)
            except AttributeError:
                # FIXME: implement API to use bilby
                raise RuntimeError(
                    "Waveform not implemented in ml4gw.waveforms"
                )
        else:
            self.approximant = approximant

        self._frequency_domain_h_plus = None
        self._frequency_domain_h_cross = None
        self._time_domain_h_plus = None
        self._time_domain_h_cross = None

    def frequency_domain_strain(self, *args, **kwargs):
        if self._frequency_domain_h_plus:
            return (
                self._frequency_domain_h_plus,
                self._frequency_domain_h_cross,
            )

        freq_mask = self.frequency_array >= self.f_min
        freq_mask *= self.frequency_array < self.f_max

        _hp, _hc = self.approximant(
            self.frequency_array[freq_mask], *args, **kwargs
        )

        self._frequency_domain_h_plus = torch.zeros(
            _hp.shape[0],
            self.frequency_array.shape[-1],
            device=self.device,
            dtype=_hp.dtype,
        )
        self._frequency_domain_h_cross = torch.zeros(
            _hc.shape[0],
            self.frequency_array.shape[-1],
            device=self.device,
            dtype=_hc.dtype,
        )
        self._frequency_domain_h_plus[..., freq_mask] = _hp
        self._frequency_domain_h_cross[..., freq_mask] = _hc
        return self._frequency_domain_h_plus, self._frequency_domain_h_cross

    def time_domain_strain(self):
        if self._frequency_domain_h_plus is None:
            raise RuntimeError("Frequency domain strain not set")
        # implemented based on https://git.ligo.org/lscsoft/bilby/-/blob/master/bilby/core/utils/series.py  # noqa
        if self._time_domain_h_plus is None:
            _hp = torch.fft.irfft(self._frequency_domain_h_plus)
            _hc = torch.fft.irfft(self._frequency_domain_h_cross)
            self._time_domain_h_plus = _hp * self.sampling_frequency
            self._time_domain_h_cross = _hc * self.sampling_frequency
        return self._time_domain_h_plus, self._time_domain_h_cross
