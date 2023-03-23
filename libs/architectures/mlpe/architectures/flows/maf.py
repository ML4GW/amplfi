from dataclasses import dataclass
from typing import Callable, Tuple

import torch
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from nflows.transforms import CompositeTransform, RandomPermutation
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
)

from mlpe.architectures.embeddings import NChannelDenseEmbedding
from mlpe.architectures.flows.flow import NormalizingFlow


@dataclass
class MaskedAutoRegressiveFlow(NormalizingFlow):
    shape: Tuple[int, int, int]
    num_transforms: int = 10
    hidden_features: int = 50
    num_blocks: int = 2
    activation: Callable = torch.tanh
    use_batch_norm: bool = False
    use_residual_blocks: bool = True

    def __post_init__(self):
        self.param_dim, self.n_ifos, self.strain_dim = self.shape
        # FIXME: port to project config; remove hardcoding
        self.embedding_net = NChannelDenseEmbedding(
            self.n_ifos,
            self.strain_dim,
            128,
            activation=self.activation,
            hidden_layer_size=256,
            num_hidden_layers=4,
        )

        super().__init__(
            self.param_dim,
            self.n_ifos,
            self.strain_dim,
            num_flow_steps=self.num_transforms,
            embedding_net=self.embedding_net,
        )

    def transform_block(self):
        """Returns the single block of the MAF"""
        single_block = [
            MaskedAffineAutoregressiveTransform(
                features=self.param_dim,
                hidden_features=self.hidden_features,
                context_features=self.context_dim,
                num_blocks=self.num_blocks,
                activation=self.activation,
                use_batch_norm=self.use_batch_norm,
                use_residual_blocks=self.use_residual_blocks,
            ),
            RandomPermutation(features=self.param_dim),
        ]
        return single_block

    def distribution(self):
        """Returns the base distribution for the flow"""
        return StandardNormal([self.param_dim])

    def build_flow(self):
        transforms = []
        for idx in range(self.num_transforms):
            transforms.extend(self.transform_block())

        transform = CompositeTransform(transforms)
        base_dist = self.distribution()
        self._flow = Flow(transform, base_dist, self.embedding_net)
