import torch
import torch.nn.functional as F
from torch import Tensor


class VICRegLoss(torch.nn.Module):
    """Implementation of the VICReg loss [0].

    This implementation is based on the code published by the authors [1].

    - [0] VICReg, 2022, https://arxiv.org/abs/2105.04906
    - [1] https://github.com/facebookresearch/vicreg/

    Attributes:
        lambda_param:
            Scaling coefficient for the invariance term of the loss.
        mu_param:
            Scaling coefficient for the variance term of the loss.
        nu_param:
            Scaling coefficient for the covariance term of the loss.
        eps:
            Epsilon for numerical stability.

    """

    def __init__(
        self,
        lambda_param: float = 25.0,
        mu_param: float = 25.0,
        nu_param: float = 1.0,
        eps: float = 0.0001,
        max_std: float = 1.0,
    ):
        """Initializes the VICRegLoss module with the specified parameters."""
        super(VICRegLoss, self).__init__()

        self.lambda_param = lambda_param
        self.mu_param = mu_param
        self.nu_param = nu_param
        self.max_std = max_std
        self.eps = eps

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Returns VICReg loss.

        Args:
            z_a:
                Tensor with shape (batch_size, ..., dim).
            z_b:
                Tensor with shape (batch_size, ..., dim).

        Returns:
            The computed VICReg loss.

        """

        # Invariance term of the loss
        inv_loss = invariance_loss(x=z_a, y=z_b)

        # Variance and covariance terms of the loss
        var_loss = 0.5 * (
            variance_loss(x=z_a, eps=self.eps, max_std=self.max_std)
            + variance_loss(x=z_b, eps=self.eps, max_std=self.max_std)
        )
        cov_loss = covariance_loss(x=z_a) + covariance_loss(x=z_b)

        # Total VICReg loss
        loss = (
            self.lambda_param * inv_loss
            + self.mu_param * var_loss
            + self.nu_param * cov_loss
        )
        return loss, (inv_loss, var_loss, cov_loss)


def invariance_loss(x: Tensor, y: Tensor) -> Tensor:
    """Returns VICReg invariance loss.

    Args:
        x:
            Tensor with shape (batch_size, ..., dim).
        y:
            Tensor with shape (batch_size, ..., dim).

    Returns:
        The computed VICReg invariance loss.
    """
    return F.mse_loss(x, y)


def variance_loss(
    x: Tensor, eps: float = 0.0001, max_std: float = 1.0
) -> Tensor:
    """Returns VICReg variance loss.

    Args:
        x:
            Tensor with shape (batch_size, ..., dim).
        eps:
            Epsilon for numerical stability.

    Returns:
        The computed VICReg variance loss.
    """
    std = torch.sqrt(x.var(dim=0) + eps)
    loss = torch.mean(F.relu(max_std - std))
    return loss


def covariance_loss(x: Tensor) -> Tensor:
    """Returns VICReg covariance loss.

    Generalized version of the covariance loss with support for tensors with more than
    two dimensions. Adapted from VICRegL:
    https://github.com/facebookresearch/VICRegL/blob/803ae4c8cd1649a820f03afb4793763e95317620/main_vicregl.py#L299 # noqa

    Args:
        x: Tensor with shape (batch_size, ..., dim).

    Returns:
          The computed VICReg covariance loss.
    """
    x = x - x.mean(dim=0)
    batch_size = x.size(0)
    dim = x.size(-1)
    # nondiag_mask has shape (dim, dim) with 1s on all non-diagonal entries.
    nondiag_mask = ~torch.eye(dim, device=x.device, dtype=torch.bool)

    # cov has shape (..., dim, dim)
    cov = torch.einsum("b...c,b...d->...cd", x, x) / (batch_size - 1)

    loss = cov[..., nondiag_mask].pow(2).sum(-1) / dim
    return loss.mean()
