import numpy as np
import pytest
import torch

from mlpe.data.dataloader import PEInMemoryDataset


@pytest.fixture
def sample_rate():
    return 512


@pytest.fixture(params=[1, 4])
def kernel_length(request):
    return request.param


@pytest.fixture(params=[8, 32])
def batch_size(request):
    return request.param


@pytest.fixture(params=[1, 10])
def batches_per_epoch(request):
    return request.param


@pytest.fixture(params=[1, 2])
def num_ifos(request):
    return request.param


@pytest.fixture
def sequential_data(num_ifos, sample_rate):
    x = np.arange(num_ifos * 128 * sample_rate)
    x = x.reshape(num_ifos, 128 * sample_rate)
    return x


@pytest.fixture(params=[True, False])
def coincident(request):
    return request.param


def test_pe_in_memory_dataloader(
    sequential_data,
    kernel_length,
    batch_size,
    batches_per_epoch,
    coincident,
    num_ifos,
    sample_rate,
):
    kernel_size = int(kernel_length * sample_rate)
    dataset = PEInMemoryDataset(
        sequential_data,
        kernel_size,
        batch_size,
        batches_per_epoch=batches_per_epoch,
        preprocessor=None,
        coincident=coincident,
        shuffle=True,
        device="cpu",
    )

    for i, X in enumerate(dataset):
        assert X.shape == (batch_size, num_ifos, kernel_size)

        for sample in X:
            start = sample[0, 0]
            expected = torch.arange(start, start + kernel_size)
            assert (sample[0] == expected).all().item()
            if num_ifos == 1:
                continue

            if coincident:
                expected = expected + 128 * sample_rate
                assert (sample[1] == expected).all().item()
            else:
                start = sample[1, 0]
                expected = torch.arange(start, start + kernel_size)
                assert (sample[1] == expected).all().item()
    assert (i + 1) == batches_per_epoch


def test_pe_in_memory_dataloader_with_preprocessor(
    sequential_data, sample_rate
):
    def preprocessor(X):
        X[:, 0] *= 2
        params = np.ones(len(X))
        return X, params

    dataset = PEInMemoryDataset(
        sequential_data,
        kernel_size=2 * sample_rate,
        batch_size=8,
        batches_per_epoch=2,
        preprocessor=preprocessor,
        coincident=True,
        shuffle=True,
        device="cpu",
    )
    for X, params in dataset:
        assert ((X[:, 0] % 2) == 0).all().item()
        if X.shape[1] > 1:
            assert not ((X[:, 1] % 2) == 0).all().item()
        assert (params == 1).all()
