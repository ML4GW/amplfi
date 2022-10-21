from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
from nflows import distributions, flows, transforms


class NormalizingFlow(ABC):
    def __init__(
        self,
        param_dim: int,
        context_dim: int,
        num_flow_steps: int,
        embedding_net: Optional[torch.nn.Module] = None,
    ):

        self.param_dim = param_dim
        self.context_dim = context_dim
        self.num_flow_steps = num_flow_steps
        self.embedding_net = embedding_net

    @abstractmethod
    def transform_block(self, idx: int) -> Callable[int, transforms.Transform]:
        pass

    @abstractmethod
    def linear_block(self) -> transforms.Transform:
        pass

    @abstractmethod
    def distribution(self) -> distributions.Distribution:
        pass

    @property
    def flow(self):
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

        flow = flows.Flow(
            self.transform,
            self.distribution(),
            embedding_net=self.embedding_net,
        )
        return flow
