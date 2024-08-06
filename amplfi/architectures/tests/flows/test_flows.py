import pytest
import torch

from amplfi.architectures.embeddings import ResNet
from amplfi.architectures.flows import (
    CouplingFlow,
    InverseAutoregressiveFlow,
    MaskedAutoregressiveFlow,
)


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
    strain = torch.randn((100, n_ifos, strain_dim))

    embedding = ResNet(n_ifos, layers=[1, 1], context_dim=context_dim)

    coupling_flow = CouplingFlow(
        param_dim,
        context_dim,
        embedding,
        num_transforms=num_transforms,
    )
    log_likelihoods = coupling_flow.log_prob(data, context=strain)

    assert log_likelihoods.shape == (len(data),)


def test_autoregressive_flow(
    param_dim, strain_dim, context_dim, n_ifos, num_transforms
):
    data = torch.randn((100, param_dim))
    strain = torch.randn((100, n_ifos, strain_dim))

    embedding = ResNet(n_ifos, layers=[1, 1], context_dim=context_dim)

    iaf = InverseAutoregressiveFlow(
        param_dim,
        context_dim,
        embedding,
        num_transforms=num_transforms,
    )
    log_likelihoods = iaf.log_prob(data, context=strain)
    assert log_likelihoods.shape == (len(data),)

    maf = MaskedAutoregressiveFlow(
        param_dim,
        context_dim,
        embedding,
        num_transforms=num_transforms,
    )
    log_likelihoods = maf.log_prob(data, context=strain)
    assert log_likelihoods.shape == (len(data),)
