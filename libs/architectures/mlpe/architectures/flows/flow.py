from abc import ABC, abstractmethod
from typing import Callable

import torch
from mlpe.architectures.embeddings import Flattener
from nflows import distributions, flows, transforms


class NormalizingFlow(ABC):
    def __init__(
        self,
        param_dim: int,
        n_ifos: int,
        strain_dim: int,
        num_flow_steps: int,
        embedding_net: torch.nn.Module = Flattener(),
    ):
        """
        Base class for normalizing flow models.
        This class should not be used directly, but instead should be subclassed.

        Args:
            param_dim: 
                The dimensionality of the parameter space for inference
            strain_dim: 
                The dimensionality of the strain data. (i.e. number of time samples)
            n_ifos: 
                The number of interferometers
            num_flow_steps:
                The number of flow blocks to use in the normalizing flow
            embedding_net:
                The embedding network for transforming strain before passing to the
                normalizing flow.
        """
        self.param_dim = param_dim
        self.strain_dim = strain_dim
        self.n_ifos = n_ifos
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
    def context_dim(self):
        sample_tensor = torch.zeros((1, self.n_ifos, self.strain_dim))
        context_dim = self.embedding_net(sample_tensor).shape[-1]
        return context_dim

    @property
    def flow(self):
        """
        Constructs the normalizing flow model.
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

        print(self.embedding_net)
        flow = flows.Flow(
            self.transform,
            self.distribution(),
            embedding_net=self.embedding_net,
        )
        return flow
