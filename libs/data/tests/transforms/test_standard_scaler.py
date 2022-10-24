import pytest
import torch
from mlpe.data.transforms.standard_scaler import StandardScalerTransform


def test_standard_scaler_transform(num_parameters):
    scaler = StandardScalerTransform(num_parameters)
    assert len(list(scaler.parameters())) == 2

    for i, param in enumerate([scaler.mean, scaler.std]):
        assert param.ndim == 1
        assert len(param) == num_parameters
        assert (param == i).all()

    x = torch.arange(10).type(torch.float32)
    X = torch.column_stack([x + i for i in range(num_parameters)])
    scaler.fit(X)

    expected_mean = torch.Tensor([4.5 + i for i in range(num_parameters)])
    assert (scaler.mean == expected_mean).all()
    assert (scaler.std == (110 / 12) ** 0.5).all()

    y = scaler(X)
    assert (y.mean(axis=0) == 0).all()
    assert (y.std(axis=0) == 1).all()

    with pytest.raises(ValueError):
        scaler.fit(torch.randn((1024)))

    # can't sub-slice the number of ifos. We could
    # add more, but I'll let the other tests check
    # for this and save ourselves some boilerplate
    if num_parameters == 1:
        return

    for bad_batch in [X[None, :, :], X[:, :1]]:
        with pytest.raises(ValueError):
            scaler(bad_batch)
