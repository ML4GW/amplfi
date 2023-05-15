from abc import ABC, abstractmethod
from typing import Callable

import torch
from nflows import distributions, transforms

from mlpe.architectures.embeddings import Flattener


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
        This class should not be used directly,
        but instead should be subclassed.

        Args:
            param_dim:
                The dimensionality of the parameter space for inference
            strain_dim:
                The dimensionality of the strain data.
                (i.e. number of time samples)
            n_ifos:
                The number of interferometers
            num_flow_steps:
                The number of flow blocks to use in the normalizing flow
            embedding_net:
                The embedding network for transforming strain
                before passing to the normalizing flow.
        """
        self._flow = None
        self.param_dim = param_dim
        self.strain_dim = strain_dim
        self.n_ifos = n_ifos
        self.num_flow_steps = num_flow_steps
        self.embedding_net = embedding_net

    @abstractmethod
    def transform_block(
        self, *args, **kwargs
    ) -> Callable[int, transforms.Transform]:
        pass

    @abstractmethod
    def distribution(self) -> distributions.Distribution:
        pass

    @abstractmethod
    def build_flow(self) -> None:
        """Initialzes flow and sets it to ``_flow`` attribute"""
        pass

    def set_weights_from_state_dict(self, state_dict):
        if self._flow is None:
            raise ValueError(
                "Flow is not built. Call build_flow() before setting weights"
            )
        self._flow.load_state_dict(state_dict)

    def to_device(self, device):
        if self._flow is None:
            raise ValueError(
                "Flow is not built. Call build_flow() before sending to device"
            )
        self._flow = self._flow.to(device)

    @property
    def context_dim(self):
        dummy_tensor = torch.zeros((1, self.n_ifos, self.strain_dim))
        context_dim = self.embedding_net(dummy_tensor).shape[-1]
        return context_dim

    @property
    def flow(self):
        if self._flow is None:
            self.build_flow()
        return self._flow
