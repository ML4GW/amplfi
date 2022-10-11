import pytest
import torch
from mlpe.architectures.embeddings import Conv1dEmbedding


@pytest.fixture(params=[1, 2, 3])
def n_ifos(request):
    return request.param


@pytest.fixture(params=[2048, 8192])
def length(request):
    return request.param


@pytest.fixture(
    params=[
        10,
        20,
    ]
)
def kernel_size(request):
    return request.param


@pytest.fixture(params=[128, 256])
def out_features(request):
    return request.param


def test_convolution(n_ifos, length, out_features, kernel_size):
    convolution = Conv1dEmbedding(n_ifos, length, out_features, kernel_size)

    x = torch.randn(8, n_ifos, length)
    y = convolution(x)

    assert y.shape == (8, out_features)
