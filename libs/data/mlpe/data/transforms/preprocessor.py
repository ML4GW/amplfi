from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch

from ml4gw.transforms import Whitening


class Preprocessor(torch.nn.Module):
    """
    Module for encoding PE preprocessing procedure.

    In the context of PE, preprocessing must be done
    on the strain (e.g. whitening) and also on the parameters
    (e.g. standardization). The forward pass of this module
    will normalize both strain and parameters.

    Args:
        num_ifos:
            Number of interferometers.
        sample_rate:
            Sample rate of the data.
        fduration:
            Duration of the time domain whitening filter.
        normalizer:
            torch.nn.Module that takes in a tensor of parameters
            of shape (n_dim, batch) and returns a transformed
            tensor of shape (n_dim, batch). If not passed, the identity
            transform is performed.
    """

    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        fduration: Optional[float] = None,
        normalizer: torch.nn.Module = torch.nn.Identity(),
    ) -> None:
        super().__init__()
        self.whitener = Whitening(
            num_ifos,
            sample_rate,
            fduration,
        )
        self.normalizer = normalizer

    def forward(self, strain: "torch.Tensor", parameters: "torch.Tensor"):
        x = self.whitener(strain)
        # since the normalizer standardizes along dim=1
        # we need to transpose the parameters tensor so that
        # we normalize along each individual attribute, and not each
        # individual sample
        parameters.transpose(0, 1)
        normed = self.normalizer(parameters)
        # transpose back to for consistency with flow.log_prob,
        normed.transpose(0, 1)

        return x, normed
