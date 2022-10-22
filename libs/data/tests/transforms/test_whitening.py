import numpy as np
import torch
from gwpy.timeseries import TimeSeries
from mlpe.data.transforms.whitening import (
    DEFAULT_FFTLENGTH,
    WhiteningTransform,
)


def test_whitening_transform_agrees_with_gwpy(
    data_length, sample_rate, num_ifos
):
    kernel_length = 4
    fduration = 1

    whitener = WhiteningTransform(
        num_ifos, sample_rate, kernel_length, highpass=0, fduration=fduration
    )
    assert len(list(whitener.parameters())) == 1
    assert whitener.time_domain_filter.ndim == 3
    assert len(whitener.time_domain_filter) == num_ifos
    assert (
        whitener.time_domain_filter.shape[-1] == (fduration * sample_rate) - 1
    )

    background = np.random.normal(
        loc=1, scale=0.5, size=(num_ifos, sample_rate * data_length)
    )
    background = torch.tensor(background)
    whitener.fit(background)

    raw = np.random.normal(
        loc=1, scale=0.5, size=(8, num_ifos, sample_rate * kernel_length)
    )
    raw = torch.Tensor(raw)

    whitened = whitener(raw)

    # now do everything with gwpy and make sure they align
    asds = []
    for ifo in background.cpu().numpy():
        ts = TimeSeries(ifo, dt=1 / sample_rate)
        asd = ts.asd(
            fftlength=DEFAULT_FFTLENGTH, method="median", window="hanning"
        )
        asds.append(asd)

    for x, y in zip(raw.cpu().numpy(), whitened.cpu().numpy()):
        for ifo, output, asd in zip(x, y, asds):
            ts = TimeSeries(ifo, dt=1 / sample_rate)
            target = ts.whiten(
                fflength=DEFAULT_FFTLENGTH,
                method="median",
                window="hanning",
                asd=asd,
                fduration=fduration,
            )
            crop_seconds = fduration / 2

            # crop timeseries for direct comparison to
            # WhiteningTransform
            target = target.crop(crop_seconds, kernel_length - crop_seconds)
            target = target.value

            # TODO: this agreement isn't very tight but I can't
            # get it consistently closer to this. Is this just
            # numerical precision issues adding up?

            # Changed this to absolute tolerance. relative tolerance
            # fails for really small numbers
            assert np.isclose(target, output, atol=1e-5).all()

    # TODO: test shape checks on fit and forward


def test_whitening_transform_produces_zero_mean_unit_variance(
    data_length, sample_rate, num_ifos
):
    kernel_length = 4

    whitener = WhiteningTransform(
        num_ifos, sample_rate, kernel_length, highpass=0, fduration=1
    )

    # generate background from random normal
    # with 0.5 std and mean 1
    background = np.random.normal(
        loc=1, scale=0.5, size=(num_ifos, sample_rate * data_length)
    )
    background = torch.Tensor(background)

    sample = np.random.normal(
        loc=1, scale=0.5, size=(8, num_ifos, sample_rate * kernel_length)
    )
    sample = torch.Tensor(sample)

    whitener.fit(background)

    whitened = whitener(sample)

    means = whitened.mean(dim=-1).numpy().flatten()
    stds = whitened.std(dim=-1).numpy().flatten()

    # is this tolerance too weak?
    assert np.isclose(means, 0, atol=1e-1).all()
    assert np.isclose(stds, 1, atol=1e-1).all()
