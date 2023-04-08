import pytest
import torch

from mlpe.architectures.embeddings import ResNet
from mlpe.architectures.flows import CouplingFlow, MaskedAutoRegressiveFlow


@pytest.fixture(params=[2, 10])
def param_dim(request):
    return request.param


@pytest.fixture(params=[512, 1024])
def strain_dim(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def n_ifos(request):
    return request.param


@pytest.fixture(params=[1, 2, 4])
def num_transforms(request):
    return request.param


def test_coupling_flow(param_dim, strain_dim, n_ifos, num_transforms):
    data = torch.randn((100, param_dim))
    strain = torch.randn((100, n_ifos, strain_dim))
    embedding = ResNet((n_ifos, None), layers=[1, 1], context_dim=10)

    coupling_flow = CouplingFlow(
        (param_dim, n_ifos, strain_dim), embedding, num_transforms
    )

    flow = coupling_flow.flow
    log_likelihoods = flow.log_prob(data, context=strain)

    assert log_likelihoods.shape == (len(data),)


def test_masked_autoregressive_flow(
    param_dim, strain_dim, n_ifos, num_transforms
):

    data = torch.randn((100, param_dim))
    strain = torch.randn((100, n_ifos, strain_dim))
    embedding = ResNet((n_ifos, None), layers=[1, 1], context_dim=10)
    maf = MaskedAutoRegressiveFlow(
        (param_dim, n_ifos, strain_dim), embedding, num_transforms
    )
    flow = maf.flow
    log_likelihoods = flow.log_prob(data, context=strain)
    assert log_likelihoods.shape == (len(data),)
