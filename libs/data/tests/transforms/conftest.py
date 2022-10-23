import pytest


@pytest.fixture(params=[1, 2, 4])
def num_ifos(request):
    return request.param


@pytest.fixture(params=[1, 2, 4])
def num_parameters(request):
    return request.param


@pytest.fixture
def data_length():
    return 128


@pytest.fixture(params=[512, 4096])
def sample_rate(request):
    return request.param
