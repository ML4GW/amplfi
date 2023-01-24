import pytest
import torch
from mlpe.architectures.flows import CouplingFlow


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
def num_flow_steps(request):
    return request.param


def test_coupling_flow(param_dim, strain_dim, n_ifos, num_flow_steps):
    data = torch.randn((100, param_dim))
    strain = torch.randn((100, n_ifos, strain_dim))

    coupling_flow = CouplingFlow(
        (param_dim, n_ifos, strain_dim), num_flow_steps
    )

    flow = coupling_flow.flow
    log_likelihoods = flow.log_prob(data, context=strain)

    assert log_likelihoods.shape == (len(data),)
