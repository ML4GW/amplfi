import pytest
import torch
from mlpe.architectures import create_flow


@pytest.fixture(params=[2, 10])
def param_dim(request):
    return request.param


@pytest.fixture(params=[100, 200])
def context_dim(request):
    return request.param


@pytest.fixture(params=[1, 2, 4])
def num_flow_steps(request):
    return request.param


def test_normalizing_flow(param_dim, context_dim, num_flow_steps):
    data = torch.randn((100, param_dim))
    context = torch.randn((100, context_dim))
    flow = create_flow(param_dim, context_dim, num_flow_steps)

    log_likelihoods = flow.log_prob(data, context=context)
    assert log_likelihoods.shape == (len(data),)
