import numpy as np
from lalinference import BurstSineGaussian, BurstSineGaussianF


def generate_time_domain_sine_gaussian(
    qualities: np.ndarray,
    frequencies: np.ndarray,
    hrss: np.ndarray,
    phases: np.ndarray,
    eccentricities: np.ndarray,
    sample_rate: float,
    duration: float,
):

    """
    Generate time domain sine-Gaussian waveforms using lalinference. See
    https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalinference/lib/LALInferenceBurstRoutines.c#L381
    for more details on parameter definitions.
    """

    delta_t = 1 / sample_rate
    samples = zip(qualities, frequencies, hrss, eccentricities, phases)
    cross, plus = [], []
    for sample in samples:

        # calculate waveform with lalsimulation
        hplus, hcross = BurstSineGaussian(*sample, delta_t=delta_t)
        hplus = hplus.data.data
        hcross = hcross.data.data

        # pad with zeros to desired waveform duration
        waveform_size = int(sample_rate * duration)

        total_pad = waveform_size - len(hplus)
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left

        hplus = np.pad(hplus, (pad_left, pad_right), mode="constant")
        hcross = np.pad(hcross, (pad_left, pad_right), mode="constant")

        cross.append(hcross)
        plus.append(hplus)

    cross = np.stack(cross)
    plus = np.stack(plus)
    return cross, plus


def bilby_frequency_domain_sine_gaussian(
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

    Meant as a wrapper to use with bilby PE.

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
    cross = np.zeros(n_freqs, dtype=complex)
    plus = np.zeros(n_freqs, dtype=complex)

    cross[: len(hcross.data.data)] = hcross.data.data
    plus[: len(hplus.data.data)] = hplus.data.data

    return dict(cross=cross, plus=plus)
