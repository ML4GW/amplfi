import torch

from amplfi.train.losses import VICRegLoss


def test_vicreg():
    batch_1 = -5 * torch.randn(1000000, 2)
    batch_2 = 5 * torch.randn(1000000, 2)

    vicreg = VICRegLoss(
        lambda_param=1, mu_param=1, nu_param=1, max_std=1, eps=1e-4
    )

    loss, (inv_loss, var_loss, cov_loss) = vicreg(batch_1, batch_2)

    # each dimension is generated with unit variance so var loss should be 0
    assert torch.isclose(var_loss, torch.tensor(0.0), atol=1e-2)

    # mse 2 * 5**2
    assert torch.isclose(inv_loss, torch.tensor(50.0), atol=1)

    # covariance should be near 0
    assert torch.isclose(cov_loss, torch.tensor(0.0), atol=1e-2)
