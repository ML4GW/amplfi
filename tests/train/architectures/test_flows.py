import pytest
import torch

from amplfi.train.architectures.embeddings import ResNet
from amplfi.train.architectures.flows import NSF


@pytest.fixture(params=[2, 10])
def param_dim(request):
    return request.param


@pytest.fixture(params=[512, 1024])
def strain_dim(request):
    return request.param


@pytest.fixture(params=[10, 12])
def context_dim(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def n_ifos(request):
    return request.param


@pytest.fixture(params=[1, 2, 4])
def num_transforms(request):
    return request.param


def test_coupling_flow(
    param_dim, strain_dim, context_dim, n_ifos, num_transforms
):
    data = torch.randn((100, param_dim))
    strain = (torch.randn((100, n_ifos, strain_dim)), None)

    embedding = ResNet(n_ifos, layers=[1, 1], context_dim=context_dim)

    flow = NSF(
        param_dim,
        embedding,
        passes=2,
        transforms=num_transforms,
    )
    log_likelihoods = flow.log_prob(data, context=strain)

    assert log_likelihoods.shape == (len(data),)


def test_autoregressive_flow(
    param_dim, strain_dim, context_dim, n_ifos, num_transforms
):
    data = torch.randn((100, param_dim))
    strain = (torch.randn((100, n_ifos, strain_dim)), None)

    embedding = ResNet(n_ifos, layers=[1, 1], context_dim=context_dim)

    flow = NSF(
        param_dim,
        embedding,
        transforms=num_transforms,
    )
    log_likelihoods = flow.log_prob(data, context=strain)
    assert log_likelihoods.shape == (len(data),)
