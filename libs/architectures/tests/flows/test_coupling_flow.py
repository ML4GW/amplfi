import pytest
import torch
from mlpe.architectures.flows import CouplingFlow


@pytest.fixture(params=[2, 10])
def param_dim(request):
    return request.param


@pytest.fixture(params=[100, 200])
def context_dim(request):
    return request.param


@pytest.fixture(params=[1, 2, 4])
def num_flow_steps(request):
    return request.param


def test_coupling_flow(param_dim, context_dim, num_flow_steps):
    data = torch.randn((100, param_dim))
    context = torch.randn((100, context_dim))

    coupling_flow = CouplingFlow((param_dim, context_dim), num_flow_steps)

    flow = coupling_flow.flow
    log_likelihoods = flow.log_prob(data, context=context)

    assert log_likelihoods.shape == (len(data),)
