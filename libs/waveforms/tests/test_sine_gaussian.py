from pathlib import Path

import bilby
import numpy as np
import pytest
from mlpe.waveforms import sine_gaussian_frequency


@pytest.fixture(params=["priors/sine_gaussian.prior"])
def prior_file(request):
    return str(Path(__file__).resolve().parent / request.param)


@pytest.fixture(params=[2048, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[2, 4, 8])
def duration(request):
    return request.param


@pytest.fixture()
def waveform_generator(duration, sample_rate):
    waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration,
        sampling_frequency=sample_rate,
        frequency_domain_source_model=sine_gaussian_frequency,
    )
    return waveform_generator


def test_sine_gaussian(waveform_generator, prior_file):
    # sample from the prior
    # and convert into desirable format
    prior = bilby.core.prior.PriorDict(prior_file)
    sample_params = prior.sample(100)
    sample_params = [
        dict(zip(sample_params, col)) for col in zip(*sample_params.values())
    ]

    # get frequencies from waveform_generator
    frequencies = waveform_generator.frequency_array

    # for each sample instance
    for p in sample_params:
        # get frequency domain strain
        frequency_strain = waveform_generator.frequency_domain_strain(p)

        # make sure maximimum frequency is
        # close enough to injected value
        hp_max_freq_arg = np.argmax(np.abs(frequency_strain["plus"]))
        hc_max_freq_arg = np.argmax(np.abs(frequency_strain["cross"]))

        assert hp_max_freq_arg == hc_max_freq_arg

        max_freq = frequencies[hp_max_freq_arg]

        assert np.isclose(max_freq, p["frequency"], rtol=1e-2)
