import torch

from typing import Literal, Optional

from ml4gw.nn.norm import NormLayer
from ml4gw.nn.resnet.resnet_1d import ResNet1D

from amplfi.architectures.embeddings.base import Embedding


class MultiModal(Embedding):
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
        **kwargs
    ):
        super().__init__()
        time_dims = (
            context_dim // 2 if context_dim % 2 == 0 else context_dim // 2 + 1
        )
        frequency_dims = context_dim // 2
        self.time_domain_resnet = ResNet1D(
            in_channels=num_ifos,
            layers=layers,
            classes=time_dims,
            kernel_size=kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )
        self.frequency_domain_resnet = ResNet1D(
            in_channels=int(num_ifos * 2),
            layers=layers,
            classes=frequency_dims,
            kernel_size=kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )

    def forward(self, X):
        time_domain_embedded = self.time_domain_resnet(X)
        X_fft = torch.fft.fft(X)
        X_fft = torch.cat((X_fft.real, X_fft.imag), dim=1)
        frequency_domain_embedded = self.frequency_domain_resnet(X_fft)

        embedding = torch.concat(
            (time_domain_embedded, frequency_domain_embedded), dim=1
        )
        return embedding
