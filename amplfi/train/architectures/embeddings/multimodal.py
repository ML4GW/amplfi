from typing import Literal, Optional

import torch
from ml4gw.nn.norm import NormLayer
from ml4gw.nn.resnet.resnet_1d import ResNet1D
from ml4gw.transforms.decimator import Decimator

from .base import Embedding


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
        **kwargs,
    ):
        """
        MultiModal embedding network that embeds both time and frequency data.

        We pass the data through their own ResNets defined by their layers
        and context dims, then concatenate the output embeddings.
        """
        super().__init__()
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

        # set the context dimension so
        # the flow can access it
        self.context_dim = time_context_dim + freq_context_dim

    def forward(self, X):
        # unpack, ignoring asds
        strain, _ = X
        time_domain_embedded = self.time_domain_resnet(strain)
        strain_fft = torch.fft.rfft(strain)
        strain_fft = torch.cat((strain_fft.real, strain_fft.imag), dim=1)
        frequency_domain_embedded = self.frequency_domain_resnet(strain_fft)

        embedding = torch.concat(
            (time_domain_embedded, frequency_domain_embedded), dim=1
        )
        return embedding


class MultiModalPsd(Embedding):
    """
    MultiModal embedding network that embeds both time and frequency data.

    We pass the data through their own ResNets defined by their layers
    and context dims, then concatenate the output embeddings.
    """

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
        **kwargs,
    ):
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

        self.freq_psd_resnet = ResNet1D(
            in_channels=int(num_ifos * 3),
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
        strain, asds = X

        asds *= 1e23
        asds = asds.float()
        inv_asds = 1 / asds

        time_domain_embedded = self.time_domain_resnet(strain)
        X_fft = torch.fft.rfft(strain)
        X_fft = X_fft[..., -asds.shape[-1] :]
        X_fft = torch.cat((X_fft.real, X_fft.imag, inv_asds), dim=1)

        frequency_domain_embedded = self.freq_psd_resnet(X_fft)
        embedding = torch.concat(
            (time_domain_embedded, frequency_domain_embedded),
            dim=1,
        )
        return embedding


class MultiModalPsdEmbeddingWithDecimator(Embedding):
    """A variant of :class:`MultiModalPsd` where the
    strain timeseries is decimated according to a schedule
    using :ref:`ml4gw.transforms.Decimator`. The frequency
    domain view(s) can be individually embedded for the
    schedule(s) if desired.
    """

    def __init__(
        self,
        num_ifos: int,
        strain_sample_rate: int,
        decimator_schedule: list,
        time_context_dim: int,
        freq_context_dim: int,
        time_layers: list[int],
        freq_layers: list[int],
        split_by_schedule: bool = True,
        time_kernel_size: int = 3,
        freq_kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
        **kwargs,
    ):
        super().__init__()
        decimator_schedule = (
            decimator_schedule
            if isinstance(decimator_schedule, torch.Tensor)
            else torch.tensor(decimator_schedule)
        )
        self.register_buffer("decimator_schedule", decimator_schedule)
        self.decimator = Decimator(
            strain_sample_rate, schedule=self.decimator_schedule
        )
        self.split_by_schedule = split_by_schedule
        self.context_dim = time_context_dim + freq_context_dim * (
            len(self.decimator_schedule) if split_by_schedule else 1
        )

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
        freq_psd_resnets = []
        for _ in range(len(decimator_schedule) if split_by_schedule else 1):
            freq_psd_resnets.append(
                ResNet1D(
                    # the number 3 is for real and imag parts of fft,
                    # and the psd arrays
                    in_channels=int(num_ifos * 3),
                    layers=freq_layers,
                    classes=freq_context_dim,
                    kernel_size=freq_kernel_size,
                    zero_init_residual=zero_init_residual,
                    groups=groups,
                    width_per_group=width_per_group,
                    stride_type=stride_type,
                    norm_layer=norm_layer,
                )
            )
        self.freq_psd_resnets = torch.nn.ModuleList(freq_psd_resnets)

    def forward(self, X):
        strain, asds = X

        asds *= 1e23
        asds = asds.float()
        inv_asds = 1 / asds

        strain = self.decimator(strain, split=False)
        strain_split = (
            self.decimator.split_by_schedule(strain)
            if self.split_by_schedule
            else [strain]
        )

        time_domain_embedded = self.time_domain_resnet(strain)

        freq_domain_embedded = []
        for _split, _resnet in zip(
            strain_split, self.freq_psd_resnets, strict=False
        ):
            _split_fft = torch.fft.rfft(_split)
            # limit to minimum of asd/fft size
            _idx = min(_split_fft.shape[-1], inv_asds.shape[-1])
            _split_fft = _split_fft[..., :_idx]
            _inv_asd = inv_asds[..., :_idx]

            _split_fft = torch.cat(
                (_split_fft.real, _split_fft.imag, _inv_asd), dim=1
            )

            _split_embedded = _resnet(_split_fft)
            freq_domain_embedded.append(_split_embedded)

        freq_domain_embedded = torch.hstack(freq_domain_embedded)

        embedding = torch.concat(
            (time_domain_embedded, freq_domain_embedded),
            dim=1,
        )

        return embedding


class FrequencyPsd(Embedding):
    """
    Single embedding for frequency domain data with ASDS
    """

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
        super().__init__()
        self.freq_psd_resnet = ResNet1D(
            in_channels=int(num_ifos * 3),
            layers=layers,
            classes=context_dim,
            kernel_size=kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )
        self.context_dim = context_dim

    def forward(self, X):
        strain, asds = X

        asds *= 1e23
        asds = asds.float()
        inv_asds = 1 / asds

        X_fft = torch.fft.rfft(strain)
        X_fft = X_fft[..., -asds.shape[-1] :]
        X_fft = torch.cat((X_fft.real, X_fft.imag, inv_asds), dim=1)
        embedding = self.freq_psd_resnet(X_fft)

        return embedding
