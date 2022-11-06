from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch
from generate_testing_set import main
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from mlpe.injection.priors import sg_uniform
from mlpe.injection.waveforms import sine_gaussian_frequency
from utils import inject_into_background


@pytest.fixture(params=[["H1", "L1"]])
def ifos(request):
    return request.param


@pytest.fixture(params=[2048, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[1, 2])
def kernel_length(request):
    return request.param


@pytest.fixture(params=[10, 50])
def n_samples(request):
    return request.param


def test_inject_into_background(ifos, sample_rate, kernel_length):

    # dummy 1000 seconds of background
    background = torch.zeros((len(ifos), sample_rate * 1000))

    n_waveforms = 100
    waveforms = []
    for i in range(len(ifos)):

        waveform = [
            np.zeros(4 * kernel_length * sample_rate) + j * (-1) ** i
            for j in range(n_waveforms)
        ]
        waveforms.append(waveform)

    waveforms = np.stack(waveforms)

    waveforms = waveforms.transpose(1, 0, 2)

    waveforms = torch.as_tensor(waveforms)

    kernel_size = int(sample_rate * kernel_length)

    injections = inject_into_background(
        background,
        waveforms,
        kernel_size,
    )

    assert injections.shape == (n_waveforms, len(ifos), kernel_size)


def test_generate_test_set(
    ifos, sample_rate, kernel_length, n_samples, datadir, logdir
):

    waveform_duration = 4

    n_background_samples = sample_rate * 1000
    background_dict = TimeSeriesDict()
    for ifo in ifos:
        background_dict[ifo] = TimeSeries(data=np.zeros(n_background_samples))

    query_patch = patch(
        "generate_testing_set.query_segments", return_value=[[0, 1000]]
    )
    download_patch = patch(
        "generate_testing_set.download_data", return_value=background_dict
    )

    with query_patch, download_patch:

        signal_file = main(
            ifos,
            None,
            None,
            None,
            None,
            None,
            sample_rate,
            sg_uniform,
            sine_gaussian_frequency,
            n_samples,
            kernel_length,
            waveform_duration,
            datadir,
            logdir,
        )

        with h5py.File(signal_file) as f:
            injections = f["injections"][:]
            assert injections.shape == (
                n_samples,
                len(ifos),
                kernel_length * sample_rate,
            )
