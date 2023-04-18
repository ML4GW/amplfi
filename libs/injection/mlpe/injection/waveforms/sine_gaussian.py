import math

import numpy as np
import torch
from lalinference import BurstSineGaussianF
from scipy.signal.windows import tukey
from torch import Tensor


def sine_gaussian_frequency(
    frequency_array: np.ndarray,
    hrss: float,
    quality: float,
    frequency: float,
    phase: float,
    eccentricity: float,
    **kwargs,
):

    """
    Frequency domain sine gaussian built on top of
    lalinference BurstSineGaussianF.
    Meant to be used in conjunction with bilby WaveformGenerator
    Args:
        frequency_array: frequencies at which to evaluate model
        hrss: amplitude
        q: quality factor
        frequency: central frequency of waveform
        phase: phase of waveform
        eccentricity: relative fraction of hplus / hcross

    """

    # number of frequencies
    n_freqs = len(frequency_array)

    # infer df from frequency array
    df = frequency_array[1] - frequency_array[0]

    # infer dt from nyquist
    dt = 1 / (2 * frequency_array[-1])

    # calculate hplus and hcross from lalinference
    hplus, hcross = BurstSineGaussianF(
        quality, frequency, hrss, eccentricity, phase, df, dt
    )
    plus = np.zeros(n_freqs, dtype=complex)
    cross = np.zeros(n_freqs, dtype=complex)

    plus[: len(hplus.data.data)] = hplus.data.data
    cross[: len(hcross.data.data)] = hcross.data.data

    return dict(plus=plus, cross=cross)


def semi_major_minor_from_e(e: Tensor):
    a = 1.0 / torch.sqrt(2.0 - e * e)
    b = a * torch.sqrt(1.0 - e * e)
    return a, b


def tukey_window(num: int, alpha: float = 0.5):
    return torch.tensor(tukey(num, alpha=alpha))


def torch_sine_gaussian(
    quality: Tensor,
    frequency: Tensor,
    hrss: Tensor,
    phase: Tensor,
    eccentricity: Tensor,
    sample_rate: float,
    duration: float,
):

    # reshape for calculating waveforms in batch
    frequency = frequency.view(-1, 1)
    quality = quality.view(-1, 1)
    hrss = hrss.view(-1, 1)
    phase = phase.view(-1, 1)
    eccentricity = eccentricity.view(-1, 1)

    pi = torch.Tensor([math.pi])

    # determine times based on requested duration and sample rate
    # and shift so that the waveform is centered at t=0
    num = int(duration * sample_rate)
    times = torch.arange(num, dtype=torch.float64) / sample_rate
    times -= duration / 2.0

    # calculate relative hplus / hcross amplitudes based on eccentricity
    # as well as normalization factors
    a, b = semi_major_minor_from_e(eccentricity)
    cosine_norm = (
        quality
        / (4.0 * frequency * torch.sqrt(pi))
        * (1.0 + torch.exp(-quality * quality))
    )
    sine_norm = (
        quality
        / (4.0 * frequency * torch.sqrt(pi))
        * (1.0 - torch.exp(-quality * quality))
    )
    cos_phase = torch.cos(phase)
    sin_phase = torch.sin(phase)
    h0_plus = (
        hrss
        * a
        / torch.sqrt(
            cosine_norm * (cos_phase**2) + sine_norm * (sin_phase**2)
        )
    )
    h0_cross = (
        hrss
        * b
        / torch.sqrt(
            cosine_norm * (sin_phase**2) + sine_norm * (cos_phase**2)
        )
    )

    # cast the phase to a complex number
    phi = 2 * pi * frequency * times
    complex_phase = torch.complex(torch.zeros_like(phi), (phi - phase))

    tukey = tukey_window(num)

    # calculate the waveform and apply a tukey window to taper the waveform
    fac = torch.exp(phi**2 / (-2.0 * quality**2) + complex_phase)
    fac *= tukey
    hplus = fac.real * h0_plus
    hcross = fac.imag * h0_cross

    return hplus, hcross
