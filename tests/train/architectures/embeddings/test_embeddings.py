from contextlib import nullcontext

import pytest
import torch

from amplfi.train.architectures.embeddings import (
    MultiModal,
    ResNet,
    TimeDomainHeterodynedEmbedding,
    MultiModalPsdEmbeddingWithDecimator,
    MultiModalHeterodynedEmbedding,
    HeterodynedEmbeddingWithDecimator,
    FrequencyDomainHeterodynedEmbedding,
)
from amplfi.train.architectures.embeddings.dense import DenseEmbedding


@pytest.fixture(params=[1, 2, 3])
def n_ifos(request):
    return request.param


@pytest.fixture(params=[2048, 8192])
def length(request):
    return request.param


@pytest.fixture(
    params=[
        3,
        5,
    ]
)
def kernel_size(request):
    return request.param


@pytest.fixture(params=[128, 256])
def out_features(request):
    return request.param


@pytest.fixture(params=[32, 64, 128])
def time_out_features(request):
    return request.param


@pytest.fixture(params=[32, 64, 128])
def freq_out_features(request):
    return request.param


def test_dense_embedding(n_ifos, length):
    embedding = DenseEmbedding(length, 10)
    x = torch.randn(8, n_ifos, length)
    y = embedding(x)

    assert y.shape == (8, n_ifos, 10)


def test_resnet(n_ifos, length, out_features, kernel_size):
    embedding = ResNet(n_ifos, out_features, [3, 3], kernel_size)
    x = torch.randn(100, n_ifos, length)
    y = embedding((x, None))
    assert y.shape == (100, out_features)


def test_multimodal(
    n_ifos, length, time_out_features, freq_out_features, kernel_size
):
    embedding = MultiModal(
        n_ifos,
        time_out_features,
        freq_out_features,
        [3, 3],
        [3, 3],
        kernel_size,
        kernel_size,
    )
    x = (torch.randn(100, n_ifos, length), None)
    y = embedding(x)
    assert y.shape == (100, time_out_features + freq_out_features)


@pytest.mark.parametrize(
    "decimator_schedule,overlapping_schedule",
    [
        [[[0, 40, 256], [40, 60, 512]], False],
        [[[0, 30, 256], [30, 50, 512], [50, 60, 2048]], False],
        [[[0, 40, 256], [20, 60, 512]], True],
    ],
)
def test_multimodal_with_decimator(
    decimator_schedule, overlapping_schedule, n_ifos, kernel_size
):
    sample_rate = 2048
    length = sample_rate * 60
    time_context_dim = 8
    freq_context_dim = 12

    ctx = (
        nullcontext()
        if not overlapping_schedule
        else pytest.raises(RuntimeError)
    )
    with ctx:
        embedding = MultiModalPsdEmbeddingWithDecimator(
            n_ifos,
            sample_rate,
            decimator_schedule,
            time_context_dim,
            freq_context_dim,
            [3, 3],
            [3, 3],
            time_kernel_size=kernel_size,
            freq_kernel_size=kernel_size,
        )
        psds = torch.randn(100, n_ifos, sample_rate)
        x = (torch.randn(100, n_ifos, length), psds)
        y = embedding(x)
        assert y.shape == (100, embedding.context_dim)


@pytest.mark.parametrize(
    "chirp_mass_low,chirp_mass_high,num_chirp_masses,chirp_mass_spacing",
    [
        (1, 2.5, 10, "log"),
        (2, 3, 10, "linear"),
    ],
)
def test_heterodyned_embedding(
    chirp_mass_low,
    chirp_mass_high,
    num_chirp_masses,
    chirp_mass_spacing,
    n_ifos,
    kernel_size,
):
    sample_rate = 2048
    timeseries_length = 10
    time_context_dim = 8
    freq_context_dim = 12
    batch_size = 10

    common_kwargs = dict(  # noqa C408
        num_ifos=n_ifos,
        strain_sample_rate=sample_rate,
        strain_kernel_length=timeseries_length,
        chirp_mass_low=chirp_mass_low,
        chirp_mass_high=chirp_mass_high,
        num_chirp_masses=num_chirp_masses,
        chirp_mass_spacing=chirp_mass_spacing,
    )

    def make_inputs():
        psds = torch.randn(batch_size, n_ifos, sample_rate)
        strain = torch.randn(
            batch_size, n_ifos, timeseries_length * sample_rate
        )
        return (strain, psds)

    # test all embeddings sequentially
    with torch.no_grad():
        # 1) time-domain only
        embedding = TimeDomainHeterodynedEmbedding(
            context_dim=time_context_dim,
            layers=[3, 3],
            kernel_size=kernel_size,
            **common_kwargs,
        )
        y = embedding(make_inputs())
        assert y.shape == (batch_size, embedding.context_dim)

        # 2) frequency-domain only
        embedding = FrequencyDomainHeterodynedEmbedding(
            context_dim=freq_context_dim,
            layers=[3, 3],
            kernel_size=kernel_size,
            **common_kwargs,
        )
        y = embedding(make_inputs())
        assert y.shape == (batch_size, embedding.context_dim)

        # 3) multimodal
        embedding = MultiModalHeterodynedEmbedding(
            time_context_dim=time_context_dim,
            freq_context_dim=freq_context_dim,
            time_layers=[3, 3],
            freq_layers=[3, 3],
            time_kernel_size=kernel_size,
            freq_kernel_size=kernel_size,
            **common_kwargs,
        )
        y = embedding(make_inputs())
        assert y.shape == (batch_size, embedding.context_dim)

        # 4) multimodal + decimator
        decimator_schedule = [[0, 6, 256], [6, 10, 1024]]
        embedding = HeterodynedEmbeddingWithDecimator(
            decimator_schedule=decimator_schedule,
            time_context_dim=time_context_dim,
            freq_context_dim=freq_context_dim,
            time_layers=[3, 3],
            freq_layers=[3, 3],
            time_kernel_size=kernel_size,
            freq_kernel_size=kernel_size,
            **common_kwargs,
        )
        y = embedding(make_inputs())
        assert y.shape == (batch_size, embedding.context_dim)
