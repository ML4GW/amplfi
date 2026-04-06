import math
from typing import Literal, List
import torch

from ml4gw.transforms import Heterodyne
from ml4gw.nn.norm import NormLayer
from ml4gw.nn.resnet.resnet_1d import ResNet1D

from .base import Embedding


class HeterodynedEmbedding(Embedding):
    """An embedding where the signal is heterodyned using
    several reference chirp mass values and then passed
    through separate ResNets for the time and frequency
    domain representations.
    """

    def __init__(
        self,
        num_ifos: int,
        strain_sample_rate: int,
        strain_kernel_length: int,
        time_context_dim: int,
        freq_context_dim: int,
        time_layers: List[int],
        freq_layers: List[int],
        chirp_mass_low: float = 1.0,
        chirp_mass_high: float = 2.5,
        num_chirp_masses: int = 100,
        chirp_mass_spacing: Literal["linear", "log"] = "linear",
        keep_last_n_seconds: float = 0.0,
        time_kernel_size: int = 3,
        freq_kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: list[Literal["stride", "dilation"]] | None = None,
        norm_layer: NormLayer | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        _chirp_mass_grid = self._create_chirp_mass_grid(
            chirp_mass_low,
            chirp_mass_high,
            num_chirp_masses,
            chirp_mass_spacing,
        )
        self.register_buffer("chirp_mass_grid", _chirp_mass_grid)

        self.heterodyne_transform = Heterodyne(
            sample_rate=strain_sample_rate,
            kernel_length=strain_kernel_length,
            chirp_mass=self.chirp_mass_grid,
            return_type="both",
        )

        self.time_domain_resnet = ResNet1D(
            in_channels=num_ifos * num_chirp_masses,
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
            in_channels=int(num_ifos * 2 * num_chirp_masses + num_ifos),  # the number 2 is for real and imag parts of fft, the number num_ifos is for the psd arrays
            layers=freq_layers,
            classes=freq_context_dim,
            kernel_size=freq_kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )
        self.keep_last_n_samples = int(keep_last_n_seconds * strain_sample_rate)
        self.context_dim = time_context_dim + freq_context_dim

    def _create_chirp_mass_grid(
        self,
        chirp_mass_low: float,
        chirp_mass_high: float,
        num_chirp_masses: int,
        chirp_mass_spacing: Literal["linear", "log"],
    ) -> torch.Tensor:
        if chirp_mass_spacing == "linear":
            return torch.linspace(
                chirp_mass_low, chirp_mass_high, num_chirp_masses
            )
        elif chirp_mass_spacing == "log":
            return torch.logspace(
                math.log10(chirp_mass_low),
                math.log10(chirp_mass_high),
                num_chirp_masses,
            )
        else:
            raise ValueError(
                f"Invalid chirp mass spacing: {chirp_mass_spacing}"
            )

    def forward(self, x):
        strain, asds = x
        asds *= 1e23
        asds = asds.float()
        inv_asds = 1 / asds

        # heterodyned time, frequency arrays have shapes (B, C, M, [T, F])
        X_heterodyned_time, X_heterodyned_freq = self.heterodyne_transform(
            strain
        )
        # for time array, reshape to (B, C*M, T) for ResNet input
        _B, _C_time, _M, _T = X_heterodyned_time.shape
        X_heterodyned_time = X_heterodyned_time.view(_B, _C_time * _M, _T)
        # optionally, for time array, restrict to last n_seconds
        if self.keep_last_n_samples > 0:
            X_heterodyned_time = X_heterodyned_time[..., -self.keep_last_n_samples:]

        # for frequency array, restrict last dimension to the length of the asd
        X_heterodyned_freq = X_heterodyned_freq[..., -asds.shape[-1] :]
        # reshape to (B, C*M, F) for ResNet input
        _B, _C_freq, _M, _F_freq = X_heterodyned_freq.shape
        X_heterodyned_freq = X_heterodyned_freq.view(_B, _C_freq * _M, _F_freq)
        # then concat the real, imag, and inv asd for the frequency domain view
        X_heterodyned_freq = torch.cat(
            (X_heterodyned_freq.real, X_heterodyned_freq.imag, inv_asds), dim=1
        )

        # pass through separate ResNets and concatenate
        time_domain_embedded = self.time_domain_resnet(X_heterodyned_time)
        frequency_domain_embedded = self.frequency_domain_resnet(
            X_heterodyned_freq
        )
        embedding = torch.concat(
            (time_domain_embedded, frequency_domain_embedded),
            dim=1,
        )
        return embedding
