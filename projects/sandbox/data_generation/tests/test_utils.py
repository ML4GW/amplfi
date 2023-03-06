import numpy as np
import pytest
from data_generation.utils import noise_from_psd


@pytest.fixture(params=[1024, 2048])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[0.5, 1])
def df(request):
    return request.param


@pytest.fixture(params=[1, 10])
def duration(request):
    return request.param


def test_noise_from_psd(
    sample_rate,
    df,
    duration,
):
    fmax = sample_rate // 2
    frequencies = np.arange(0, fmax + df, df)
    psd = np.ones(len(frequencies)) * 1e-46

    noise = noise_from_psd(psd, df, duration, sample_rate)

    assert len(noise) == int(duration * sample_rate)
