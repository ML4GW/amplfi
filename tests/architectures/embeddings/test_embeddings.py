import pytest
import torch

from amplfi.train.architectures.embeddings import MultiModal, ResNet
from amplfi.train.architectures.embeddings.dense import DenseEmbedding


@pytest.fixture(params=[1, 2, 3])
def n_ifos(request):
    return request.param


@pytest.fixture(params=[2048, 8192])
def length(request):
    return request.param


@pytest.fixture(
    params=[
        3,
        5,
    ]
)
def kernel_size(request):
    return request.param


@pytest.fixture(params=[128, 256])
def out_features(request):
    return request.param


@pytest.fixture(params=[32, 64, 128])
def time_out_features(request):
    return request.param


@pytest.fixture(params=[32, 64, 128])
def freq_out_features(request):
    return request.param


def test_dense_embedding(n_ifos, length):
    embedding = DenseEmbedding(length, 10)
    x = torch.randn(8, n_ifos, length)
    y = embedding(x)

    assert y.shape == (8, n_ifos, 10)


def test_resnet(n_ifos, length, out_features, kernel_size):
    embedding = ResNet(n_ifos, out_features, [3, 3], kernel_size)
    x = torch.randn(100, n_ifos, length)
    y = embedding((x, None))
    assert y.shape == (100, out_features)


def test_multimodal(
    n_ifos, length, time_out_features, freq_out_features, kernel_size
):
    embedding = MultiModal(
        n_ifos,
        time_out_features,
        freq_out_features,
        [3, 3],
        [3, 3],
        kernel_size,
        kernel_size,
    )
    x = (torch.randn(100, n_ifos, length), None)
    y = embedding(x)
    assert y.shape == (100, time_out_features + freq_out_features)
