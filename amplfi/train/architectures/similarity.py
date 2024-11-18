import torch
import torch.nn as nn

from amplfi.train.architectures.embeddings.base import Embedding


class Expander(nn.Module):
    """Projection head used for VICReg.

    "The projector network has three linear layers, each with 8192 output
    units. The first two layers of the projector are followed by a batch
    normalization layer and rectified linear units." [0]

    - [0]: 2022, VICReg, https://arxiv.org/pdf/2105.04906.pdf
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 8192,
        output_dim: int = 8192,
        num_layers: int = 3,
    ):
        """Initializes the VICRegProjectionHead with the specified dimensions.

        Args:
            input_dim:
                Dimensionality of the input features.
            hidden_dim:
                Dimensionality of the hidden layers.
            output_dim:
                Dimensionality of the output features.
            num_layers:
                Number of layers in the projection head.
        """
        super().__init__()
        blocks = [
            (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU())
            for _ in range(num_layers - 2)  # Exclude first and last layer.
        ]
        blocks = [
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            *blocks,
            (hidden_dim, output_dim, None, None),
        ]

        layers = []
        for block in blocks:
            input_dim, output_dim, batch_norm, non_linearity, *bias = block
            use_bias = bias[0] if bias else not bool(batch_norm)
            layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
            if batch_norm is not None:
                layers.append(batch_norm)
            if non_linearity is not None:
                layers.append(non_linearity)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes one forward pass through the projection head.

        Args:
            x:
                Input of shape bsz x num_ftrs.
        """
        projection: torch.Tensor = self.layers(x)
        return projection


class SimilarityEmbedding(torch.nn.Module):
    """
    Embedding network for similarity learning

    Combines an `embedding` with an expander layer
    as done in https://arxiv.org/pdf/2407.19048
    """

    def __init__(self, embedding: Embedding, expander_factor: int = 4):
        super().__init__()
        self.embedding = embedding
        self.head = Expander(
            input_dim=self.embedding.context_dim,
            hidden_dim=self.embedding.context_dim * expander_factor,
            output_dim=self.embedding.context_dim * expander_factor,
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.head(x)
        return x
