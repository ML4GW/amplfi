import numpy as np
import pytest
import torch
from data_generation.utils import inject_into_random_background, noise_from_psd


@pytest.fixture(params=[1, 2])
def kernel_length(request):
    return request.param


@pytest.fixture(params=[["H1", "L1"]])
def ifos(request):
    return request.param


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

    # inverse fourier transform noise, and check for
    # consistent power as psd
    from scipy.fft import fft

    num_samples = len(noise)
    dt = duration / num_samples
    noise_fft = fft(noise) * dt
    noise_fft_abs = np.abs(noise_fft) ** 2

    assert np.log10(noise_fft_abs).mean() == pytest.approx(-46, abs=1)


def test_inject_into_random_background(ifos, sample_rate, kernel_length):

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
    injections = inject_into_random_background(
        background,
        waveforms,
        kernel_size,
    )

    assert injections.shape == (n_waveforms, len(ifos), kernel_size)
