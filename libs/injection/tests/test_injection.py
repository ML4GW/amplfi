#!/usr/bin/env python
# coding: utf-8
from pathlib import Path

import bilby
import pytest

import mlpe.injection

TEST_DIR = Path(__file__).resolve().parent


@pytest.fixture(params=[10, 100])
def n_samples(request):
    return request.param


@pytest.fixture(params=[4])
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
def prior_file(request):
    return str(Path(__file__).resolve().parent / request.param)


def test_generate_time_domain_sine_gaussian(
    prior_file, waveform_duration, sample_rate, n_samples
):

    waveform_size = sample_rate * waveform_duration
    params = bilby.gw.prior.PriorDict(prior_file).sample(n_samples)

    cross, plus = mlpe.injection.generate_time_domain_sine_gaussian(
        frequencies=params["frequency"],
        hrss=params["hrss"],
        qualities=params["quality"],
        phases=params["phase"],
        eccentricities=params["eccentricity"],
        sample_rate=sample_rate,
        duration=waveform_duration,
    )

    assert cross.shape == (n_samples, waveform_size)
    assert plus.shape == (n_samples, waveform_size)
