#!/usr/bin/env python
# coding: utf-8
from pathlib import Path

import bilby
import mlpe.injection
import mlpe.injection.waveforms as waveforms
import pytest
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters

TEST_DIR = Path(__file__).resolve().parent


@pytest.fixture(params=[10, 100])
def n_samples(request):
    return request.param


@pytest.fixture(params=[1, 2, 4])
def waveform_duration(request):
    return request.param


@pytest.fixture(params=[2048, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture(
    params=[
        "priors/sine_gaussian.prior",
    ]
)
def sg_prior_file(request):
    return str(Path(__file__).resolve().parent / request.param)


@pytest.fixture(params=["priors/nonspin_BBH.prior"])
def cbc_prior_file(request):
    return str(Path(__file__).resolve().parent / request.param)


def test_generate_gw_sg(
    sg_prior_file, waveform_duration, sample_rate, n_samples
):

    # first test custom sine gaussian
    n_pols = 2
    waveform_size = sample_rate * waveform_duration
    sample_params = bilby.gw.prior.PriorDict(sg_prior_file).sample(n_samples)

    signals = mlpe.injection.generate_gw(
        sample_params,
        sample_rate,
        waveform_duration,
        waveform=waveforms.sine_gaussian_frequency,
    )

    expected_signal_shape = (n_samples, n_pols, waveform_size)
    assert signals.shape == expected_signal_shape


def test_generate_gw_cbc(
    cbc_prior_file, waveform_duration, sample_rate, n_samples
):

    # first test custom sine gaussian
    n_pols = 2
    waveform_size = sample_rate * waveform_duration
    sample_params = bilby.gw.prior.PriorDict(cbc_prior_file).sample(n_samples)

    waveform_arguments = dict(
        reference_frequency=20,
        minimum_frequency=30,
        waveform_approximant="IMRPhenomPv2",
    )
    signals = mlpe.injection.generate_gw(
        sample_params,
        sample_rate,
        waveform_duration,
        waveform=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=waveform_arguments,
        parameter_conversion=convert_to_lal_binary_black_hole_parameters,
    )

    expected_signal_shape = (n_samples, n_pols, waveform_size)
    assert signals.shape == expected_signal_shape
