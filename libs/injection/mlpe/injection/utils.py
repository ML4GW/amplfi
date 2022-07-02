import numpy as np
import scipy.signal as sig


def calc_snr(data, noise_psd, fs, fmin=20):
    """Calculate the waveform SNR given the background noise PSD

    Args:
        data: timeseries of the signal whose SNR is to be calculated
        noise_psd: PSD of the background that the signal is in
        fs: sampling frequency of the signal and background
        fmin: minimum frequency for the highpass filter

    Returns:
        The SNR of the signal, a single value

    """

    data_fd = np.fft.rfft(data) / fs
    data_freq = np.fft.rfftfreq(len(data)) * fs
    dfreq = data_freq[1] - data_freq[0]

    noise_psd_interp = noise_psd.interpolate(dfreq)
    noise_psd_interp[noise_psd_interp == 0] = 1.0

    snr = 4 * np.abs(data_fd) ** 2 / noise_psd_interp.value * dfreq
    snr = np.sum(snr[fmin <= data_freq])
    snr = np.sqrt(snr)

    return snr


def apply_high_pass_filter(
    signals: np.ndarray, minimum_frequency: float, sampling_frequency: float
):
    sos = sig.butter(
        N=8,
        Wn=minimum_frequency,
        btype="highpass",
        output="sos",
        fs=sampling_frequency,
    )
    filtered_signals = np.empty_like(signals)
    for i, signal in enumerate(signals):

        filtered = sig.sosfiltfilt(sos, signal, axis=1)
        filtered_signals[i] = filtered

    return filtered_signals
