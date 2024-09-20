from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Optional

import torch
from pyro.distributions import ConditionalTransformedDistribution
from pyro.distributions.conditional import ConditionalComposeTransformModule
from pyro.distributions.transforms import BatchNorm, Transform
from pyro.nn import PyroModule


class FlowArchitecture(PyroModule):
    def __init__(
        self,
        num_params: int,
        context_dim: int,
        embedding_net: torch.nn.Module,
        embedding_weights: Optional[Path] = None,
        freeze_embedding: bool = False,
        use_batch_norm: bool = True,
    ):

        super().__init__()
        self.num_params = num_params
        self.context_dim = context_dim
        self.embedding_net = embedding_net
        self.use_batch_norm = use_batch_norm

        if freeze_embedding:
            self.embedding_context = torch.no_grad
        else:
            self.embedding_context = nullcontext

        if embedding_weights is not None:
            self.embedding_net.load_state_dict(torch.load(embedding_weights))

    def transform_block(self, *args, **kwargs) -> Callable[[int], Transform]:
        raise NotImplementedError

    def distribution(self) -> torch.distributions.Distribution:
        raise NotImplementedError

    def build_transforms(self) -> ConditionalComposeTransformModule:
        transforms = []
        for _ in range(self.num_transforms):
            transforms.extend([self.transform_block()])
            if self.use_batch_norm:
                transforms.extend([BatchNorm(self.num_params)])
        return ConditionalComposeTransformModule(transforms)

    def flow(self) -> ConditionalTransformedDistribution:
        return ConditionalTransformedDistribution(
            self.distribution(), self.transforms
        )

    def log_prob(self, x, context):
        """Wrapper around :meth:`log_prob` from
        `TransformedDistribution` object.
        """
        if not hasattr(self, "transforms"):
            raise RuntimeError("Flow is not built")

        with self.embedding_context():
            embedded_context = self.embedding_net(context)
        return self.flow().condition(embedded_context).log_prob(x)

    def sample(self, n, context):
        """Wrapper around :meth:`sample` from
        `TransformedDistribution` object.
        """
        if not hasattr(self, "transforms"):
            raise RuntimeError("Flow is not built")
        embedded_context = self.embedding_net(context)
        n = [n] if isinstance(n, int) else n
        return self.flow().condition(embedded_context).sample(n)
