from unittest.mock import patch

import h5py
import numpy as np
import pytest
from data_generation.flow_injections import main
from gwpy.timeseries import TimeSeries, TimeSeriesDict

from mlpe.injection.priors import sg_uniform


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


def test_flow_injections(
    ifos, sample_rate, kernel_length, n_samples, datadir, logdir
):

    waveform_duration = 4

    n_background_samples = sample_rate * 1000
    background_dict = TimeSeriesDict()
    for ifo in ifos:
        background_dict[ifo] = TimeSeries(data=np.zeros(n_background_samples))

    query_patch = patch(
        "data_generation.flow_injections.query_segments",
        return_value=[[0, 1000]],
    )
    download_patch = patch(
        "data_generation.flow_injections.download_data",
        return_value=background_dict,
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
