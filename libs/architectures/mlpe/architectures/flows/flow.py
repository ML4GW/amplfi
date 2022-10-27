from abc import ABC, abstractmethod
from typing import Callable

import torch
from mlpe.architectures.embeddings import Flattener
from nflows import distributions, flows, transforms


class NormalizingFlow(ABC):
    def __init__(
        self,
        param_dim: int,
        context_dim: int,
        num_flow_steps: int,
        embedding_net: torch.nn.Module = Flattener(),
    ):

        self.param_dim = param_dim
        self.context_dim = context_dim
        self.num_flow_steps = num_flow_steps
        self.embedding_net = embedding_net

        self.flow = self.construct_flow()

    @abstractmethod
    def transform_block(self, idx: int) -> Callable[int, transforms.Transform]:
        pass

    @abstractmethod
    def linear_block(self) -> transforms.Transform:
        pass

    @abstractmethod
    def distribution(self) -> distributions.Distribution:
        pass

    def construct_flow(self):
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

    def forward(self, n_samples: int, context):
        # set the forward method of the
        # flow to correspond to sampling
        # so we can export the model later with tensorrt

        return self.flow.sample(n_samples, context)
