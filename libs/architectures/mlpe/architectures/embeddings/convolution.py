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


class DenseEmbedding(torch.nn.Module):
    """Fully connected embedding with some hidden layers."""

    def __init__(
        self,
        in_features,
        out_features,
        hidden_layer_size=100,
        num_hidden_layers=3,
        activation=torch.relu,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.out_features = out_features
        self.activation = activation

        self.initial_layer = nn.Linear(
            self.in_features, self.hidden_layer_size
        )
        self.final_layer = nn.Linear(self.hidden_layer_size, self.out_features)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
                for _ in range(self.num_hidden_layers)
            ]
        )

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.activation(x)
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.final_layer(x)
        return x


class NChannelDenseEmbedding(nn.Module):
    """
    DenseEmbedding for N channels. Creates a :meth:`DenseEmbedding`
    for individual channel, with a final Linear layer acting on stacked
    embedding on a single channel.
    Expect input shape is (num_batch, N, in_shape). Output shape
    is (num_batch, 1, out_shape)
    """

    def __init__(
        self,
        N,
        in_shape,
        out_shape,
        **kwargs,
    ) -> None:
        super().__init__()
        self.N = N
        self.activation = kwargs.get("activation", torch.relu)
        self.embeddings = nn.ModuleList(
            [DenseEmbedding(in_shape, out_shape, **kwargs) for _ in range(N)]
        )
        self.final_layer = nn.Linear(N * out_shape, out_shape)

    def forward(self, x):
        embedded_vals = []
        for channel_num, embedding in enumerate(self.embeddings):
            embedded_vals.append(embedding(x[:, channel_num, :]))

        x_concat = torch.concat(embedded_vals, dim=1)
        x_concat = self.final_layer(x_concat)
        x_concat = self.activation(x_concat)
        return x_concat
