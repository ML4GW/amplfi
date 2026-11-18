import math
from typing import Literal, List
import torch

from ml4gw.transforms import Heterodyne
from ml4gw.nn.norm import NormLayer
from ml4gw.nn.resnet.resnet_1d import ResNet1D
from ml4gw.transforms.decimator import Decimator
import torch.nn.functional as F

from .base import Embedding


class _HeterodyneTransformMixin:
    """Heterodyne transform setup."""

    def _setup_heterodyne_transform(
        self,
        strain_sample_rate: int,
        strain_kernel_length: int,
        chirp_mass_low: float,
        chirp_mass_high: float,
        num_chirp_masses: int,
        chirp_mass_spacing: Literal["linear", "log"],
    ) -> None:
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

def select_top_k(
    X: torch.Tensor,
    top_k: int,
    kernel_size: int = 31,
    stride: int = 1,
    padding: int = 15,
    keep_last_n_samples: int | None = None,
) -> torch.Tensor:
    """
    Select top `k` chirp mass channels from `M` heterodyned timeseries.

    Args:
        X (Tensor):
            Input tensor of shape (B, C, M, T) where B is the batch
            size, C is the number of channels, M is the number of chirp
            mass channels, and T is the number of time samples.
        top_k (int):
            Number of chirp mass channels to retain.
        kernel_size (int):
            Size of the running average (average pooling) window used to
            smooth the absolute value of the input before computing the
            selection statistic.
        stride (int):
            Stride of the average pooling operation.
        padding (int):
            Zero-padding applied to both sides of the timeseries before
            the average pooling operation.
        keep_last_n_samples (int, optional):
            If provided, only the final `n` samples are used when
            computing the statistic used to select the top k chirp mass
            channels. If `None`, all samples are used.

    Returns:
        Tensor:
            Output tensor of shape (B, C * top_k, T) containing the
            selected top k chirp mass channels for each batch and channel.
    """

    B, C, M, T = X.shape

    avgpool = torch.stack(
        [
            F.avg_pool1d(
                torch.abs(x.reshape(C * M, T)),
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            for x in X
        ]
    ).reshape(B, C, M, T)

    if keep_last_n_samples is not None:
        avgpool_snr = torch.sqrt(
            (avgpool[..., -keep_last_n_samples:] ** 2).sum(dim=1)
        )
    else:
        avgpool_snr = torch.sqrt((avgpool**2).sum(dim=1))

    vals = avgpool_snr.max(dim=-1).values
    idx = torch.topk(vals, k=top_k, dim=-1).indices
    idx = idx[:, None, :, None].expand(B, C, top_k, T)
    return torch.gather(X, dim=2, index=idx).reshape(B, C * top_k, T)


class TimeDomainHeterodynedFrequencyDomainDecimator(
    _HeterodyneTransformMixin, Embedding
):
    """An embedding where the signal is heterodyned in the time-domain.
    It is also decimated and the individual parts are used in Frequency-domain
    resnets.
    """

    def __init__(
        self,
        num_ifos: int,
        strain_sample_rate: int,
        decimator_schedule: list,
        strain_kernel_length: int,
        time_context_dim: int,
        freq_context_dim: int,
        time_layers: List[int],
        freq_layers: List[int],
        chirp_mass_low: float = 1.0,
        chirp_mass_high: float = 2.5,
        num_chirp_masses: int = 100,
        chirp_mass_spacing: Literal["linear", "log"] = "log",
        keep_last_n_seconds: float = 0.0,
        top_k: int = None,
        time_kernel_size: int = 3,
        freq_kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: list[Literal["stride", "dilation"]] | None = None,
        norm_layer: NormLayer | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        decimator_schedule = (
            decimator_schedule
            if isinstance(decimator_schedule, torch.Tensor)
            else torch.tensor(decimator_schedule)
        )

        self.register_buffer("decimator_schedule", decimator_schedule)
        self.decimator = Decimator(
            strain_sample_rate, schedule=self.decimator_schedule, split=True
        )
        self.context_dim = time_context_dim + freq_context_dim * len(
            self.decimator_schedule
        )
        
        self.top_k = top_k

        self._setup_heterodyne_transform(
            strain_sample_rate=strain_sample_rate,
            strain_kernel_length=strain_kernel_length,
            chirp_mass_low=chirp_mass_low,
            chirp_mass_high=chirp_mass_high,
            num_chirp_masses=num_chirp_masses,
            chirp_mass_spacing=chirp_mass_spacing,
        )

        num_time_channels = self.top_k if self.top_k is not None else num_chirp_masses

        self.time_domain_resnet = ResNet1D(
            in_channels=num_ifos * num_time_channels,
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
        for _ in range(len(decimator_schedule)):
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
        self.keep_last_n_samples = int(
            keep_last_n_seconds * strain_sample_rate
        )
        self.context_dim = time_context_dim + freq_context_dim * len(
            self.decimator_schedule
        )

    def forward(self, x):
        strain, asds = x
        asds *= 1e23
        asds = asds.float()
        inv_asds = 1 / asds

        # Handle the time domain first
        # heterodyned time, frequency arrays have shapes (B, C, M, [T, F])
        X_heterodyned_time, _ = self.heterodyne_transform(strain)
        if self.top_k is not None:
            X_heterodyned_time = select_top_k(
                X_heterodyned_time, self.top_k, keep_last_n_samples=self.keep_last_n_samples
            )
        else:
            # for time array, reshape to (B, C*M, T) for ResNet input
            _B, _C_time, _M, _T = X_heterodyned_time.shape
            X_heterodyned_time = X_heterodyned_time.view(_B, _C_time * _M, _T)
        # optionally, for time array, restrict to last n_seconds
        if self.keep_last_n_samples > 0:
            X_heterodyned_time = X_heterodyned_time[
                ..., -self.keep_last_n_samples :
            ]
        time_domain_embedded = self.time_domain_resnet(X_heterodyned_time)

        # Now handle the frequency domain
        strain_split = self.decimator(strain)
        # pass each split through its own freq domain resnet
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


class TimeDomainHeterodynedEmbedding(_HeterodyneTransformMixin, Embedding):
    """An embedding where the signal is heterodyned using
    several reference chirp mass values and then passed
    through a ResNet for the time domain representation.
    """

    def __init__(
        self,
        num_ifos: int,
        strain_sample_rate: int,
        strain_kernel_length: int,
        context_dim: int,
        layers: List[int],
        chirp_mass_low: float = 1.0,
        chirp_mass_high: float = 2.5,
        num_chirp_masses: int = 100,
        chirp_mass_spacing: Literal["linear", "log"] = "log",
        keep_last_n_seconds: float = 0.0,
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: list[Literal["stride", "dilation"]] | None = None,
        norm_layer: NormLayer | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self._setup_heterodyne_transform(
            strain_sample_rate=strain_sample_rate,
            strain_kernel_length=strain_kernel_length,
            chirp_mass_low=chirp_mass_low,
            chirp_mass_high=chirp_mass_high,
            num_chirp_masses=num_chirp_masses,
            chirp_mass_spacing=chirp_mass_spacing,
        )

        self.time_domain_resnet = ResNet1D(
            in_channels=num_ifos * num_chirp_masses,
            layers=layers,
            classes=context_dim,
            kernel_size=kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )

        self.keep_last_n_samples = int(
            keep_last_n_seconds * strain_sample_rate
        )
        self.context_dim = context_dim

    def forward(self, x):
        strain, _ = x  # ignore asd
        X_heterodyned_time, _ = self.heterodyne_transform(strain)

        # reshape to (B, C*M, T) for ResNet input
        _B, _C_time, _M, _T = X_heterodyned_time.shape
        X_heterodyned_time = X_heterodyned_time.view(_B, _C_time * _M, _T)

        # optionally restrict to last n_seconds
        if self.keep_last_n_samples > 0:
            X_heterodyned_time = X_heterodyned_time[
                ..., -self.keep_last_n_samples :
            ]

        return self.time_domain_resnet(X_heterodyned_time)


class FrequencyDomainHeterodynedEmbedding(
    _HeterodyneTransformMixin, Embedding
):
    """An embedding where the signal is heterodyned using several
    reference chirp mass values and only the frequency-domain view
    is passed through a ResNet.
    """

    def __init__(
        self,
        num_ifos: int,
        strain_sample_rate: int,
        strain_kernel_length: int,
        context_dim: int,
        layers: List[int],
        chirp_mass_low: float = 1.0,
        chirp_mass_high: float = 2.5,
        num_chirp_masses: int = 100,
        chirp_mass_spacing: Literal["linear", "log"] = "linear",
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: list[Literal["stride", "dilation"]] | None = None,
        norm_layer: NormLayer | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._setup_heterodyne_transform(
            strain_sample_rate=strain_sample_rate,
            strain_kernel_length=strain_kernel_length,
            chirp_mass_low=chirp_mass_low,
            chirp_mass_high=chirp_mass_high,
            num_chirp_masses=num_chirp_masses,
            chirp_mass_spacing=chirp_mass_spacing,
        )

        self.frequency_domain_resnet = ResNet1D(
            in_channels=int(
                num_ifos * 2 * num_chirp_masses + num_ifos
            ),  # 2 is for real and imag parts of fft, add num_ifos for asd
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

    def forward(self, x):
        strain, asds = x
        asds *= 1e23
        asds = asds.float()
        inv_asds = 1 / asds

        _, X_heterodyned_freq = self.heterodyne_transform(strain)

        # restrict last dimension to ASD length
        X_heterodyned_freq = X_heterodyned_freq[..., -asds.shape[-1] :]

        # reshape to (B, C*M, F) for ResNet input
        _B, _C_freq, _M, _F_freq = X_heterodyned_freq.shape
        X_heterodyned_freq = X_heterodyned_freq.view(_B, _C_freq * _M, _F_freq)

        # concat real, imag, and inv asd channels
        X_heterodyned_freq = torch.cat(
            (X_heterodyned_freq.real, X_heterodyned_freq.imag, inv_asds), dim=1
        )

        return self.frequency_domain_resnet(X_heterodyned_freq)


class MultiModalHeterodynedEmbedding(_HeterodyneTransformMixin, Embedding):
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

        self._setup_heterodyne_transform(
            strain_sample_rate=strain_sample_rate,
            strain_kernel_length=strain_kernel_length,
            chirp_mass_low=chirp_mass_low,
            chirp_mass_high=chirp_mass_high,
            num_chirp_masses=num_chirp_masses,
            chirp_mass_spacing=chirp_mass_spacing,
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
            in_channels=int(
                num_ifos * 2 * num_chirp_masses + num_ifos
            ),  # 2 is for real and imag parts of fft, add num_ifos for asd
            layers=freq_layers,
            classes=freq_context_dim,
            kernel_size=freq_kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )
        self.keep_last_n_samples = int(
            keep_last_n_seconds * strain_sample_rate
        )
        self.context_dim = time_context_dim + freq_context_dim

    def forward_impl(self, x):
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
            X_heterodyned_time = X_heterodyned_time[
                ..., -self.keep_last_n_samples :
            ]

        # for frequency array, restrict last dimension to the length of the asd
        X_heterodyned_freq = X_heterodyned_freq[..., -asds.shape[-1] :]
        # reshape to (B, C*M, F) for ResNet input
        _B, _C_freq, _M, _F_freq = X_heterodyned_freq.shape
        X_heterodyned_freq = X_heterodyned_freq.view(_B, _C_freq * _M, _F_freq)
        # then concat the real, imag, and inv asd for the frequency domain view
        X_heterodyned_freq = torch.cat(
            (X_heterodyned_freq.real, X_heterodyned_freq.imag, inv_asds), dim=1
        )

        return X_heterodyned_time, X_heterodyned_freq

    def forward(self, x):
        X_heterodyned_time, X_heterodyned_freq = self.forward_impl(x)
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


class HeterodynedEmbeddingWithDecimator(MultiModalHeterodynedEmbedding):
    """Same as MultiModalHeterodynedEmbedding but with an additional decimation
    step before passing to the time domain resnet.
    """

    def __init__(
        self,
        decimator_schedule: list,
        **kwargs,
    ):
        super().__init__(**kwargs)
        strain_sample_rate = kwargs["strain_sample_rate"]
        decimator_schedule = (
            decimator_schedule
            if isinstance(decimator_schedule, torch.Tensor)
            else torch.tensor(decimator_schedule)
        )
        self.register_buffer("decimator_schedule", decimator_schedule)
        self.decimator = Decimator(
            strain_sample_rate, schedule=self.decimator_schedule, split=False
        )
        # TODO: handle keep_last_n_seconds properly with decimation

    def forward(self, x):
        X_heterodyned_time, X_heterodyned_freq = self.forward_impl(x)
        # decimate the time and frequency domain embeddings separately
        X_decimated_time = self.decimator(X_heterodyned_time)
        # pass through separate ResNets and concatenate
        time_domain_embedded = self.time_domain_resnet(X_decimated_time)
        frequency_domain_embedded = self.frequency_domain_resnet(
            X_heterodyned_freq
        )
        embedding = torch.concat(
            (time_domain_embedded, frequency_domain_embedded),
            dim=1,
        )
        return embedding
