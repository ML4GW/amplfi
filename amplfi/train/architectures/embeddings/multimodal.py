from typing import Literal, Optional

import torch
from ml4gw.nn.norm import NormLayer
from ml4gw.nn.resnet.resnet_1d import ResNet1D

from amplfi.architectures.embeddings.base import Embedding


class MultiModal(Embedding):
    def __init__(
        self,
        num_ifos: int,
        time_context_dim: int,
        freq_context_dim: int,
        time_layers: list[int],
        freq_layers: list[int],
        time_kernel_size: int = 3,
        freq_kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
        **kwargs
    ):
        """
        MultiModal embedding network that embeds both time and frequency data.

        We pass the data through their own ResNets defined by their layers
        and context dims, then concatenate the output embeddings.
        """
        super().__init__()
        self.context_dim = time_context_dim + freq_context_dim
        self.time_domain_resnet = ResNet1D(
            in_channels=num_ifos,
            layers=time_layers,
            classes=time_context_dim,
            kernel_size=time_kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )
        self.frequency_domain_resnet = ResNet1D(
            in_channels=int(num_ifos * 2),
            layers=freq_layers,
            classes=freq_context_dim,
            kernel_size=freq_kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )

    def forward(self, X):
        time_domain_embedded = self.time_domain_resnet(X)
        X_fft = torch.fft.rfft(X)
        X_fft = torch.cat((X_fft.real, X_fft.imag), dim=1)
        frequency_domain_embedded = self.frequency_domain_resnet(X_fft)

        embedding = torch.concat(
            (time_domain_embedded, frequency_domain_embedded), dim=1
        )
        return embedding
