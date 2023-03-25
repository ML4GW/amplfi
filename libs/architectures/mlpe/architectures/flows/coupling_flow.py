from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import nflows.nn.nets as nn_
import torch
from nflows import distributions, transforms, utils
from nflows.flows import Flow

from mlpe.architectures.flows.flow import NormalizingFlow


@dataclass
class CouplingFlow(NormalizingFlow):
    shape: Tuple[int, int, int]
    embedding_net: torch.nn.Module
    num_transforms: int
    hidden_dim: int = 512
    num_transform_blocks: int = 2
    dropout_probability: float = 0.0
    activation: Callable = torch.nn.functional.relu
    use_batch_norm: bool = False
    num_bins: int = 8
    tails: str = "linear"
    tail_bound: float = 1.0
    apply_unconditional_transform: bool = False

    def __post_init__(self):
        # unpack shape parameters
        self.param_dim, self.n_ifos, self.strain_dim = self.shape

        super().__init__(
            self.param_dim,
            self.n_ifos,
            self.strain_dim,
            self.num_transforms,
            self.embedding_net,
        )

    def transform_block(self, idx: int):
        transform = transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=utils.create_alternating_binary_mask(
                self.param_dim, even=(idx % 2 == 0)
            ),
            # TODO: generalize the ability to specify
            # different Callables here?
            transform_net_create_fn=(
                lambda in_features, out_features: nn_.ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=self.hidden_dim,
                    context_features=self.context_dim,
                    num_blocks=self.num_transform_blocks,
                    activation=self.activation,
                    dropout_probability=self.dropout_probability,
                    use_batch_norm=self.use_batch_norm,
                )
            ),
            num_bins=self.num_bins,
            tails=self.tails,
            tail_bound=self.tail_bound,
            apply_unconditional_transform=self.apply_unconditional_transform,
        )

        return transform

    def linear_block(self):
        return transforms.CompositeTransform(
            [
                transforms.RandomPermutation(features=self.param_dim),
                transforms.LULinear(self.param_dim, identity_init=True),
            ]
        )

    def distribution(self):
        return distributions.StandardNormal((self.param_dim,))

    def build_flow(self, state_dict: Optional[Path] = None):
        """
        Constructs the normalizing flow model
        """
        self.transform = transforms.CompositeTransform(
            [
                transforms.CompositeTransform(
                    [self.transform_block(i), self.linear_block()]
                )
                for i in range(self.num_flow_steps)
            ]
            + [self.linear_block()]
        )

        flow = Flow(
            self.transform,
            self.distribution(),
            embedding_net=self.embedding_net,
        )
        if state_dict:
            flow.load_state_dict(state_dict)
        self._flow = flow
