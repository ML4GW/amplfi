import pytest
import torch

from amplfi.train.architectures.embeddings import (
    MultiModal,
    ResNet,
    MultiModalPsdEmbeddingWithDecimator,
)
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


@pytest.fixture(params=[True, False])
def split_by_schedule(request):
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


def test_multimodal_with_decimator(n_ifos, kernel_size, split_by_schedule):
    sample_rate = 2048
    length = sample_rate * 60
    decimator_schedule = torch.tensor(
        [[0, 40, 256], [40, 58, 512], [58, 60, 2048]],
        dtype=torch.int,
    )
    time_context_dim = 8
    freq_context_dim = 12

    embedding = MultiModalPsdEmbeddingWithDecimator(
        n_ifos,
        sample_rate,
        decimator_schedule,
        time_context_dim,
        freq_context_dim,
        [3, 3],
        [3, 3],
        split_by_schedule=split_by_schedule,
        time_kernel_size=kernel_size,
        freq_kernel_size=kernel_size,
    )
    psds = torch.randn(100, n_ifos, sample_rate)
    x = (torch.randn(100, n_ifos, length), psds)

    y = embedding(x)
    assert y.shape == (100, embedding.context_dim)
