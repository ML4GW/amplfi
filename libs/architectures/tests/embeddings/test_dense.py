import pytest
import torch

from mlpe.architectures.embeddings import DenseEmbedding


@pytest.fixture(params=[1, 2, 3])
def n_ifos(request):
    return request.param


@pytest.fixture(params=[2048, 8192])
def in_features(request):
    return request.param


@pytest.fixture(params=[128, 256])
def out_features(request):
    return request.param


def test_dense_embedding(n_ifos, in_features, out_features):
    shape = (n_ifos, in_features)
    embedding = DenseEmbedding(
        shape, out_features, hidden_layer_size=128, num_hidden_layers=3
    )
    x = torch.randn(8, n_ifos, in_features)
    y = embedding(x)
    assert y.shape == (8, out_features)
