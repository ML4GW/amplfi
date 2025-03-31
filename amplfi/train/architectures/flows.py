import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Sequence

import torch
import zuko

from .embeddings.base import Embedding


class FlowArchitecture(torch.nn.Module):
    """
    Base class for normalizing flow architectures that
    provides interface for interacting with embedding networks
    """

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
                k.removeprefix("model.embedding"): v
                for k, v in state_dict.items()
                if k.startswith("model.embedding")
            }
            self.embedding_net.load_state_dict(state_dict)

    def build_flow(self) -> zuko.lazy.Flow:
        raise NotImplementedError

    def log_prob(self, x, context):
        """Wrapper around :meth:`log_prob` from
        `zuko.lazy.Flow` object.
        """
        if not hasattr(self, "flow"):
            raise RuntimeError("Flow is not built")

        with self.embedding_context():
            embedded_context = self.embedding_net(context)
        return self.flow(embedded_context).log_prob(x)

    def sample(self, n, context):
        """Wrapper around :meth:`sample` from
        `TransformedDistribution` object.
        """
        if not hasattr(self, "flow"):
            raise RuntimeError("Flow is not built")
        embedded_context = self.embedding_net(context)
        return self.flow(embedded_context).sample((n,))


class NSF(FlowArchitecture):
    """
    Light wrapper around the `NSF` flow from `zuko` library
    for compatibility with the `FlowArchitecture` interface.

    See https://zuko.readthedocs.io/stable/api/zuko.flows.spline.html#zuko.flows.spline.NSF

    Args:
        transforms:
            Number of transformations in the flow
        hidden_features:
            Sequence of integers representing hidden units in the hyper network
        passes:
            Default of `None` corresponds to fully autoregressive flow.
            A value of 2 corresponds to coupling flow.
        bins:
            Number of bins in the spline
        randperm:
            Whether to randomly permute features in between
            transformation layers
        residual:
            Whether to use residual connections in the hyper network.

    """  # noqa E501

    def __init__(
        self,
        *args,
        transforms: int,
        hidden_features: Optional[Sequence[int]] = (64, 64),
        passes: Optional[int] = None,
        bins: Optional[int] = 8,
        randperm: Optional[bool] = False,
        residual: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.transforms = transforms
        self.bins = bins
        self.randperm = randperm
        self.hidden_features = hidden_features
        self.residual = residual
        self.passes = passes
        self.flow = self.build_flow()

    def build_flow(self) -> zuko.lazy.Flow:
        return zuko.flows.NSF(
            self.num_params,
            transforms=self.transforms,
            context=self.embedding_net.context_dim,
            bins=self.bins,
            randperm=self.randperm,
            hidden_features=self.hidden_features,
            passes=self.passes,
            residual=self.residual,
        )
