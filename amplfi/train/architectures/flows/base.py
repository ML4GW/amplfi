import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Optional

import torch
from pyro.distributions import ConditionalTransformedDistribution, transforms
from pyro.distributions.conditional import ConditionalComposeTransformModule
from pyro.nn import PyroModule

from ..embeddings.base import Embedding


class FlowArchitecture(PyroModule):
    def __init__(
        self,
        num_params: int,
        embedding_net: Embedding,
        embedding_weights: Optional[Path] = None,
        freeze_embedding: bool = False,
    ):

        super().__init__()
        self.num_params = num_params
        self.embedding_net = embedding_net

        if freeze_embedding:
            self.embedding_context = torch.no_grad
        else:
            self.embedding_context = nullcontext

        if embedding_weights is not None:
            # load in pre trained embedding weights,
            # removing extra module weights (like, e.g. the standard scaler)
            logging.info(f"Loading embedding weights from {embedding_weights}")
            checkpoint = torch.load(embedding_weights)
            state_dict = checkpoint["state_dict"]
            state_dict = {
                k.strip("model.embedding"): v
                for k, v in state_dict.items()
                if k.startswith("model.embedding")
            }
            self.embedding_net.load_state_dict(state_dict)

    def transform_block(
        self, *args, **kwargs
    ) -> Callable[[int], transforms.Transform]:
        raise NotImplementedError

    def distribution(self) -> torch.distributions.Distribution:
        raise NotImplementedError

    def build_transforms(self) -> ConditionalComposeTransformModule:
        """Sets the ``transforms`` attribute"""
        raise NotImplementedError

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
