from abc import ABC, abstractmethod
from typing import Callable

import torch
from pyro.distributions import ConditionalTransformedDistribution, transforms


class NormalizingFlow(ABC):
    @abstractmethod
    def transform_block(
        self, *args, **kwargs
    ) -> Callable[int, transforms.Transform]:
        pass

    @abstractmethod
    def distribution(self) -> torch.distributions.Distribution:
        pass

    @abstractmethod
    def build_flow(self) -> None:
        """Sets the ``transforms`` attribute"""
        pass

    @property
    def context_dim(self):
        dummy_tensor = torch.zeros(
            (1, self.n_ifos, self.strain_dim), device=self.device
        )
        _context_dim = self.embedding_net(dummy_tensor).shape[-1]
        return _context_dim

    @property
    def flow(self):
        return ConditionalTransformedDistribution(
            self.distribution(), self.transforms
        )

    def log_prob(self, x, context):
        """Wrapper around :meth:`log_prob` from
        `TransformedDistribution` object.
        """
        if not hasattr(self, "transforms"):
            raise RuntimeError("Flow is not built")

        embedded_context = self.embedding_net(context)
        return self.flow.condition(embedded_context).log_prob(x)

    def sample(self, n, context):
        """Wrapper around :meth:`sample` from
        `TransformedDistribution` object.
        """
        if not hasattr(self, "transforms"):
            raise RuntimeError("Flow is not built")

        embedded_context = self.embedding_net(context)
        n = [n] if isinstance(n, int) else n
        return self.flow.condition(embedded_context).sample(n)
