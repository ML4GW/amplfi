from typing import Iterable, Tuple

import gwdatafind
import lal
import lalsimulation
import numpy as np
import torch
from gwpy.timeseries import TimeSeries, TimeSeriesDict

from ml4gw.utils.slicing import slice_kernels


def noise_from_psd(
    psd: np.ndarray,
    df: float,
    duration: float,
    sample_rate: float,
):
    """
    Generate noise from a given PSD. See
    https://pycbc.org/pycbc/latest/html/_modules/pycbc/noise/gaussian.html#noise_from_psd
    for inspiration.

    Args:
        psd: np.ndarray of the PSD to generate noise from
        df: frequency resolution of `psd` in Hz
        duration: duration of the noise to generate in seconds
        sample_rate: sample rate of the noise to generate in Hz
    """

    # calculate length of requested timeseries in samples
    # and create empty array to store data
    length = int(duration * sample_rate)
    noise_ts = np.empty(length)

    # create epcoh and random seed
    epoch = lal.LIGOTimeGPS(0, 0)
    rng = lal.gsl_rng("ranlux", 0)

    # calculate number of samples we'll generate
    # per segment, and initialize the segment
    delta_t = 1 / sample_rate
    N = int(1 / delta_t / df)
    stride = N // 2
    segment = lal.CreateREAL8TimeSeries(
        "", epoch, 0.0, 1.0 / sample_rate, lal.StrainUnit, N
    )

    psd_lal = lal.CreateREAL8FrequencySeries(
        "", epoch, 0.0, df, lal.SecondUnit, len(psd)
    )
    psd_lal.data.data[:] = psd
    psd_lal.data.data[0] = 0
    psd_lal.data.data[-1] = 0

    length_generated = 0
    lalsimulation.SimNoise(segment, 0, psd_lal, rng)
    while length_generated < length:
        if (length_generated + stride) < length:
            noise_ts[
                length_generated : length_generated + stride
            ] = segment.data.data[0:stride]
        else:
            noise_ts[length_generated:length] = segment.data.data[
                0 : length - length_generated
            ]

        length_generated += stride
        lalsimulation.SimNoise(segment, stride, psd_lal, rng)
    return noise_ts


def download_data(
    ifos: Iterable[str],
    frame_type: str,
    channel: str,
    sample_rate: float,
    start: float,
    stop: float,
) -> TimeSeriesDict:
    data = TimeSeriesDict()
    for ifo in ifos:
        files = gwdatafind.find_urls(
            site=ifo.strip("1"),
            frametype=f"{ifo}_{frame_type}",
            gpsstart=start,
            gpsend=stop,
            urltype="file",
        )
        data[ifo] = TimeSeries.read(
            files, channel=f"{ifo}:{channel}", start=start, end=stop, nproc=4
        )
    return data.resample(sample_rate)


def inject_into_random_background(
    background: np.ndarray,
    waveforms: np.ndarray,
    kernel_size: int,
):
    """
    Inject waveforms into randomly sampled kernels of background data.
    The background data is sampled non-coincidentally between detectors

    Args:
        background:
            A (n_ifos, n_samples) array of background data
        waveforms:
            A (n_waveforms, n_ifos, n_samples) array of waveforms to inject
        kernel_size:
            Size of the kernels to produce

    Returns array of size (n_waveforms, n_ifos, n_samples) of injections
    """

    # select the kernel size around the center
    # of the waveforms
    center = waveforms.shape[-1] // 2
    start = center - (kernel_size // 2)
    stop = center + (kernel_size // 2)
    waveforms = waveforms[:, :, start:stop]

    # sample random, non-coincident background kernels
    num_kernels = len(waveforms)

    idx = torch.randint(
        num_kernels,
        size=(num_kernels, len(background)),
    )

    X = slice_kernels(background, idx, kernel_size)

    # inject waveforms into background
    X += waveforms

    return X


def inject_into_background(
    background: Tuple[np.ndarray, np.ndarray],
    waveforms: np.ndarray,
    signal_times: np.ndarray,
) -> np.ndarray:

    """
    Inject a set of signals into background data at specific times
    Args:
        background:
            A tuple (t, data) of np.ndarray arrays.
            The first tuple is an array of times.
            The second tuple is the background strain values
            sampled at those times.
        waveforms:
            An np.ndarary of shape (n_waveforms, waveform_size)
            that contains the waveforms to inject
        signal_times: np.ndarray,:
            An array of times where signals will be injected. Corresponds to
            first sample of waveforms.
    Returns
        A dictionary where the key is an interferometer and the value
        is a timeseries with the signals injected
    """

    times, data = background[0].copy(), background[1].copy()
    if len(times) != len(data):
        raise ValueError(
            "The times and background arrays must be the same length"
        )

    sample_rate = 1 / (times[1] - times[0])
    # create matrix of indices of waveform_size for each waveform
    num_waveforms, waveform_size = waveforms.shape
    idx = np.arange(waveform_size)[None] - int(waveform_size // 2)
    idx = np.repeat(idx, len(waveforms), axis=0)

    # offset the indices of each waveform corresponding to their time offset
    time_diffs = signal_times - times[0]
    idx_diffs = (time_diffs * sample_rate).astype("int64")
    idx += idx_diffs[:, None]

    # flatten these indices and the signals out to 1D
    # and then add them in-place all at once
    idx = idx.reshape(-1)
    waveforms = waveforms.reshape(-1)
    data[idx] += waveforms

    return data
