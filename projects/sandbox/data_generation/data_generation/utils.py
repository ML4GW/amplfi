from typing import Iterable

import gwdatafind
import numpy as np
import torch
from gwpy.timeseries import TimeSeries, TimeSeriesDict

from ml4gw.utils.slicing import slice_kernels


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


def inject_into_background(
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
