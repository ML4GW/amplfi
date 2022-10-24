from typing import Callable, Optional

import torch
from mlpe.data.transforms.whitening import WhiteningTransform


class Preprocessor(torch.nn.Module):
    """
    Module for encoding PE preprocessing procedure.

    In the context of PE, preprocessing must be done
    on the strain (e.g. whitening) and also on the parameters
    (e.g. standardization). The forward pass of this module
    will normalize both strain and parameters.

    Args:
        normalizer: Callable that normalizes parameters. If None,
        will return parameters as is.
    """

    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        normalizer: Optional[Callable] = None,
        fduration: Optional[float] = None,
        highpass: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.whitener = WhiteningTransform(
            num_ifos,
            sample_rate,
            kernel_length,
            highpass=highpass,
            fduration=fduration,
        )

        self.normalizer = normalizer

    def forward(self, strain, parameters):
        x = self.whitener(strain)
        if self.normalizer is not None:
            normed = self.normalizer(parameters)
        return x, normed
