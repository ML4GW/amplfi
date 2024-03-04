import h5py
import pytest
from data_generation.waveforms import main

from mlpe.injection.priors import sg_uniform


@pytest.fixture(params=[sg_uniform])
def prior(request):
    return request.param


@pytest.fixture(params=[4])
def waveform_duration(request):
    return request.param


@pytest.fixture(params=[2048, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[10, 100])
def n_samples(request):
    return request.param


def test_generate_waveforms(
    prior,
    sample_rate,
    n_samples,
    waveform_duration,
    datadir,
    logdir,
):

    signal_file = main(
        prior,
        sample_rate,
        n_samples,
        waveform_duration,
        datadir,
        logdir,
    )

    waveform_size = sample_rate * waveform_duration

    with h5py.File(signal_file) as f:
        for key in f.keys():
            if key == "signals":
                act_shape = f[key].shape
                exp_shape = (n_samples, 2, waveform_size)
                assert (
                    act_shape == exp_shape
                ), f"Expected shape {exp_shape} for signals, found {act_shape}"
            else:
                act_shape = f[key].shape
                exp_shape = (n_samples,)
                assert (
                    act_shape == exp_shape
                ), f"Expected shape {exp_shape} for {key}, found {act_shape}"
