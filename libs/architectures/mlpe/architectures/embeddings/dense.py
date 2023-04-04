import torch
import torch.nn as nn


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
