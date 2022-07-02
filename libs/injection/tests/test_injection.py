#!/usr/bin/env python
# coding: utf-8
from pathlib import Path

import bilby
import mlpe.injection
import mlpe.injection.waveforms as waveforms
import pytest
from bilby.gw.source import lal_binary_black_hole

TEST_DIR = Path(__file__).resolve().parent


@pytest.fixture(params=["time", "frequency"])
def domain(request):
    return request.param


@pytest.fixture(params=[1, 2, 4])
def waveform_duration(request):
    return request.param


@pytest.fixture(params=[2048, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[10, 32])
def flow(request):
    return request.param


@pytest.fixture(params=["IMRPhenomPv2", "TaylorF2"])
def approximant(request):
    return request.param


@pytest.fixture(params=["priors/nonspin_BBH.prior"])
def cbc_prior_file(request):
    return str(Path(__file__).resolve().parent / request.param)


@pytest.fixture(params=["priors/sine_gaussian.prior"])
def sg_prior_file(request):
    return str(Path(__file__).resolve().parent / request.param)


@pytest.fixture
def cbc_generator(flow, approximant, waveform_duration, sample_rate):
    waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
        duration=waveform_duration,
        sampling_frequency=sample_rate,
        frequency_domain_source_model=lal_binary_black_hole,
        waveform_arguments={
            "waveform_approximant": approximant,
            "reference_frequency": 50,
            "minimum_frequency": flow,
        },
    )
    return waveform_generator


@pytest.fixture
def sg_generator(waveform_duration, sample_rate):
    waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
        duration=waveform_duration,
        sampling_frequency=sample_rate,
        frequency_domain_source_model=waveforms.sine_gaussian_frequency,
    )

    return waveform_generator


def test_generate_gw(cbc_prior_file, domain, cbc_generator):
    """Test generate_gw using supplied waveform generator, or
    initializing generator
    """

    duration = cbc_generator.duration
    sample_rate = cbc_generator.sampling_frequency

    sample_params = bilby.gw.prior.BBHPriorDict(cbc_prior_file).sample(10)

    signals = mlpe.injection.generate_gw(
        sample_params,
        domain,
        cbc_generator,
    )

    if domain == "time":
        waveform_size = duration * sample_rate

    elif domain == "frequency":
        df = 1 / duration
        fmax = sample_rate / 2
        waveform_size = int(fmax / df) + 1

    expected_signal_shape = (10, 2, waveform_size)
    assert signals.shape == expected_signal_shape


def test_generate_gw_sg(sg_prior_file, domain, sg_generator):
    """Test generate_gw using supplied waveform generator, or
    initializing generator
    """

    duration = sg_generator.duration
    sample_rate = sg_generator.sampling_frequency

    sample_params = bilby.gw.prior.PriorDict(sg_prior_file).sample(10)

    signals = mlpe.injection.generate_gw(
        sample_params,
        domain,
        sg_generator,
    )

    if domain == "time":
        waveform_size = duration * sample_rate

    elif domain == "frequency":
        df = 1 / duration
        fmax = sample_rate / 2
        waveform_size = int(fmax / df) + 1

    expected_signal_shape = (10, 2, waveform_size)
    assert signals.shape == expected_signal_shape
