import torch
from lightly.models.modules.heads import VICRegProjectionHead

from amplfi.train.architectures.embeddings.base import Embedding


class SimilarityEmbedding(torch.nn.Module):
    """
    Embedding network for similarity learning

    Combines an `embedding` with an expander layer
    as done in https://arxiv.org/pdf/2407.19048
    """

    def __init__(self, embedding: Embedding, expander_factor: int = 4):
        super().__init__()
        self.embedding = embedding
        self.head = VICRegProjectionHead(
            input_dim=self.embedding.context_dim,
            hidden_dim=self.embedding.context_dim * expander_factor,
            output_dim=self.embedding.context_dim * expander_factor,
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.head(x)
        return x
