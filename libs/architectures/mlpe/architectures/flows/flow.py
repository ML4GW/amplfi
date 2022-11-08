from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
from mlpe.architectures.embeddings import Flattener
from nflows import distributions, flows, transforms


class NormalizingFlow(ABC):
    def __init__(
        self,
        param_dim: int,
        context_dim: int,
        num_flow_steps: int,
    ):

        self.param_dim = param_dim
        self.context_dim = context_dim
        self.num_flow_steps = num_flow_steps

    @abstractmethod
    def transform_block(self, idx: int) -> Callable[int, transforms.Transform]:
        pass

    @abstractmethod
    def linear_block(self) -> transforms.Transform:
        pass

    @abstractmethod
    def distribution(self) -> distributions.Distribution:
        pass

    def construct_flow(
        self,
        context_preprocessor: Optional[torch.nn.Module] = None,
        embedding_net: torch.nn.Module = Flattener(),
    ):
        """
        Constructs the normalizing flow model.

        Args:
            context_preprocessor:
                Torch module that will be used
                to preprocess the context (i.e. strain)
            embedding_net:
                Torch module for compressing context before passing to the flow
        """

        # construct the flow transform
        self.transform = transforms.CompositeTransform(
            [
                transforms.CompositeTransform(
                    [self.transform_block(i), self.linear_block()]
                )
                for i in range(self.num_flow_steps)
            ]
            + [self.linear_block()]
        )

        # construct the embedding_net as a sequential operation
        # of the context pre processor,
        # and then whatever embedding network is passed
        if context_preprocessor is not None:
            embedding_net = torch.nn.Sequential(
                context_preprocessor, embedding_net
            )

        flow = flows.Flow(
            self.transform,
            self.distribution(),
            embedding_net=embedding_net,
        )

        # override forward method so sampling can be exported
        flow.forward = self.forward
        return flow

    def forward(self, n_samples, context):
        # set the forward method of the
        # flow to correspond to sampling
        # so we can export the model later with tensorrt

        # TODO: to hack around this issue in hermes:
        # https://github.com/ML4GW/hermes/issues/36
        # we pass a dummy tensor, and use its length
        # to infer the number of samples

        n_samples = len(n_samples)

        return self.flow.sample(n_samples, context)
