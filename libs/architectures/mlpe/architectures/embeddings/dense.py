from typing import Callable, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class BasicBlock(torch.nn.Module):
    def __init__(
        self,
        size: int,
        n_channels: int,
        activation: Callable = torch.relu,
    ):
        super().__init__()
        self.activation = activation
        self.layer = nn.Linear(size, size)
        self.norm = nn.BatchNorm1d(n_channels)

    def forward(self, x: Tensor) -> Tensor:

        out = self.layer(x)
        # out = self.norm(out)
        out = self.activation(out)

        return out


class DenseEmbedding(torch.nn.Module):
    """Fully connected embedding

    Embeds a Tensor of shape (num_batch, n_ifos, n_samples)
    into a Tensor of shape (num_batch, 1, out_features)
    through a fully connected network with num_hidden_layers
    hidden layers of size hidden_layer_size.

    After passing through the hidden layers, the tensor is passed
    through a penultimate layer that will output a Tensor of shape
    (num_batch, n_ifos, out_features). Finally, the tensor is concated along
    the ifo dimension and passed through a final layer that
    will output a Tensor of shape (num_batch, 1, out_features).


    Args:
        shape: A (n_ifos, in_features) Tuple
        context_dim: Number of total output features
        hidden_layer_size: Number of hidden units in each hidden layer
        num_hidden_layers: Number of hidden layers
        activation: Activation function to use between layers
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        context_dim: int,
        hidden_layer_size: int,
        num_hidden_layers: int,
    ) -> None:
        super().__init__()
        n_ifos, in_features = shape
        # TODO: make activation a parameter; typeo was having trouble here
        self.activation = torch.nn.functional.relu
        self.initial_layer = nn.Linear(in_features, hidden_layer_size)
        self.hidden_layers = nn.Sequential(
            *[
                BasicBlock(
                    hidden_layer_size, n_ifos, activation=self.activation
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.fc1 = nn.Linear(hidden_layer_size, context_dim)
        self.fc2 = nn.Linear(n_ifos * context_dim, context_dim)

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.activation(x)
        x = self.hidden_layers(x)
        x = self.fc1(x)
        x = x.reshape(len(x), 1, -1)
        x = self.fc2(x)
        x = x.reshape(len(x), -1)
        return x
