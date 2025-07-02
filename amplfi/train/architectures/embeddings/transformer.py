import math

import torch
import torch.nn as nn
from torch import Tensor
from .base import Embedding


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe
        return x


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        self.pe = nn.Parameter(
            torch.empty(1, d_model, max_len).normal_(std=0.02)
        )

    def forward(self, x):
        return x + self.pe


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=3,
            bias=True,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.GELU()
        self.conv_last = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x.transpose(2, 1)).transpose(2, 1)
        x = self.act(x)
        x = self.conv_last(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        num_ifos: int,
        out_dim: int,
        d_model: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_layers: int = 18,
        prenorm: bool = True,
        max_len: int = 1365,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.num_ifos = num_ifos
        self.dropout = dropout
        self.out_dim = out_dim

        self.conv_proj = nn.Conv1d(
            in_channels=num_ifos,
            out_channels=d_model,
            kernel_size=3,
            stride=3,
            bias=False,
        )
        self.avgpool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)

        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
            norm_first=prenorm,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=self.num_layers
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.d_model, self.out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_proj(x)
        x = self.avgpool(x)
        x = self.pos_encoder(x)

        # rearrange data dormat to (batch, seq, features)
        x = x.permute(0, 2, 1)
        x = self.transformer_encoder(x)

        # restore data dormat to (batch, features, seq)
        x = x.permute(0, 2, 1)
        # pooling and flatten
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class TransformerMultiModalPsd(Embedding):
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
        time_layers: int,
        freq_layers: int,
        **kwargs,
    ):
        super().__init__()
        self.context_dim = time_context_dim + freq_context_dim
        self.time_domain_transformer = Transformer(
            num_ifos,
            time_context_dim,
            d_model=128,
            num_heads=4,
            dropout=0.0,
            num_layers=time_layers,
            max_len=1365,
        )
        self.freq_domain_transformer = Transformer(
            (num_ifos * 3),
            freq_context_dim,
            d_model=128,
            num_heads=4,
            dropout=0.0,
            num_layers=freq_layers,
            max_len=673,
        )

    def forward(self, X):
        strain, asds = X

        asds *= 1e23
        asds = asds.float()
        inv_asds = 1 / asds

        time_domain_embedded = self.time_domain_transformer(strain)
        X_fft = torch.fft.rfft(strain)
        X_fft = X_fft[..., -asds.shape[-1] :]
        X_fft = torch.cat((X_fft.real, X_fft.imag, inv_asds), dim=1)

        frequency_domain_embedded = self.freq_domain_transformer(X_fft)
        embedding = torch.concat(
            (time_domain_embedded, frequency_domain_embedded),
            dim=1,
        )
        return embedding
