import pytest
import torch
from mlpe.architectures import coupling_flow


@pytest.fixture(params=[2, 10])
def param_dim(request):
    return request.param


@pytest.fixture(params=[100, 200])
def context_dim(request):
    return request.param


def test_normalizing_flow(param_dim, context_dim):
    data = torch.randn((100, param_dim))
    context = torch.randn((100, context_dim))
    flow = coupling_flow(param_dim, context_dim)

    log_likelihoods = flow.log_prob(data, context=context)
    assert log_likelihoods.shape == (len(data),)
