import numpy as np
from lalinference import BurstSineGaussianF


def sine_gaussian_frequency(
    frequency_array: np.ndarray
    hrss: float,
    q: float,
    frequency: float,
    phase: float,
    eccentricity: float,
):
    
    """
    Frequency domain sine gaussian built on top of lalinference BurstSineGaussianF
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
        q, frequency, hrss, eccentricity, phase, df, dt
    )
    plus = np.zeros(len(n_freqs), dtype=complex)
    cross = np.zeros(len(n_freqs), dtype=complex)

    plus[: len(hplus.data.data)] = hplus.data.data
    cross[: len(hcross.data.data)] = hcross.data.data

    return dict(plus=plus, cross=cross)
