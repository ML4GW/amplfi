from unittest.mock import MagicMock

import pytest
import torch

from mlpe.data.transforms import PEInjector


@pytest.fixture(params=[["H1", "L1"]])
def ifos(request):
    return request.param


# here we only test the forward call
# as the rest is tested in ml4gw
def test_waveform_injector(ifos):

    # create background of all zeros
    # and a magic mock for the WaveformInjector`1`2
    background = torch.zeros((128, 2, 2048))
    mock_injector = MagicMock()

    # create a bunch of random waveforms
    waveforms = torch.randn(128, 2, 4096)

    # mock sample call to return all the waveforms
    # in order and dummy params
    def mock_sample(N):
        return waveforms[:N], None

    mock_injector.sample_waveforms = mock_sample

    # call forward method with injector and background
    X, params = PEInjector.forward(mock_injector, background)

    # assert that the center of waveform
    # timseries is in correct location
    center = X.shape[-1] // 2
    waveform_center = waveforms.shape[-1] // 2
    for i, data in enumerate(X):
        for j in range(len(ifos)):
            assert data[j][center] == waveforms[i][j][waveform_center]
