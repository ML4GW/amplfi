from typing import Literal, Optional

from ml4gw.nn.norm import NormLayer
from ml4gw.nn.resnet.resnet_1d import ResNet1D

from .base import Embedding


class ResNet(ResNet1D, Embedding):
    def __init__(
        self,
        num_ifos: int,
        context_dim: int,
        layers: list[int],
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
    ):

        super().__init__(
            num_ifos,
            layers=layers,
            classes=context_dim,
            kernel_size=kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )

        # set the context dimension so
        # the flow can access it
        self.context_dim = context_dim

    def forward(self, x):
        strain, _ = x
        return super().forward(strain)
