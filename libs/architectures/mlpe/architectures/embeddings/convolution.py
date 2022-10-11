# 1 dimensional convolutional encoder


import torch
import torch.nn as nn


class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        activation: nn.Module = nn.Tanh,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = activation()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class FCBlock(torch.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, dropout_rate: float = 0.5
    ):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        return self.fc(x)


class Conv1dEmbedding(torch.nn.Module):
    def __init__(
        self, in_channels: int, size: int, out_features: int, kernel_size: int
    ):
        super().__init__()

        self.in_channels = in_channels
        self.size = size

        self.downsampler = nn.Sequential()

        input_conv = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=10,
            stride=1,
            padding=3,
        )
        self.downsampler.add_module("INPUT_CONV", input_conv)

        for i, out_channels in enumerate([8, 16, 32, 64]):
            conv_block = ConvBlock(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=3,
            )
            self.downsampler.add_module(f"CONV_{i+1}", conv_block)
            in_channels = out_channels

        fc_dim = self.get_flattened_size(size)

        self.fc = FCBlock(fc_dim, out_features)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.downsampler(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

    def get_flattened_size(self, size: int):
        """
        Return flattened size after convolutional encoding
        """
        x = torch.rand(1, self.in_channels, size)

        out = self.downsampler(x)
        out = torch.flatten(out)
        return len(out)
