from pathlib import Path

import h5py
import mlpe.injection.waveforms as waveforms
import pytest
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.source import lal_binary_black_hole
from generate_waveforms import main


@pytest.fixture(params=["priors/nonspin_BBH.prior"])
def cbc_prior_file(request):
    return str(Path(__file__).resolve().parent / request.param)


@pytest.fixture(params=["priors/sine_gaussian.prior"])
def sg_prior_file(request):
    return str(Path(__file__).resolve().parent / request.param)


@pytest.fixture(params=[1, 2, 4])
def waveform_duration(request):
    return request.param


@pytest.fixture(params=[2048, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[10, 100])
def n_samples(request):
    return request.param


def test_generate_waveforms_custom_model(
    sg_prior_file,
    sample_rate,
    n_samples,
    waveform_duration,
    datadir,
    logdir,
):

    signal_file = main(
        sg_prior_file,
        waveforms.sine_gaussian_frequency,
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


def test_generate_waveforms_cbc_model(
    cbc_prior_file,
    sample_rate,
    n_samples,
    waveform_duration,
    datadir,
    logdir,
):

    signal_file = main(
        cbc_prior_file,
        lal_binary_black_hole,
        sample_rate,
        n_samples,
        waveform_duration,
        datadir,
        logdir,
        waveform_arguments={
            "waveform_approximant": "IMRPhenomPv2",
            "reference_frequency": 50,
            "minimum_frequency": 20,
        },
        parameter_conversion=convert_to_lal_binary_black_hole_parameters,
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
