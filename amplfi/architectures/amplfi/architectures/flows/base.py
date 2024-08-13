from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Optional

import torch
from ml4gw.transforms import ChannelWiseScaler
from pyro.distributions import ConditionalTransformedDistribution, transforms
from pyro.distributions.conditional import ConditionalComposeTransformModule
from pyro.nn import PyroModule

class FlowArchitecture(PyroModule):
    def __init__(
        self,
        num_params: int,
        context_dim: int,
        embedding_net: torch.nn.Module,
        embedding_weights: Optional[Path] = None,
        freeze_embedding: bool = False,
    ):

        super().__init__()
        self.num_params = num_params
        self.context_dim = context_dim
        self.embedding_net = embedding_net

        if freeze_embedding:
            self.embedding_context = torch.no_grad
        else:
            self.embedding_context = nullcontext

        def load_weights(weights: Path):
            checkpoint = torch.load(weights)
            arch_weights = {
                k[6:]: v
                for k, v in checkpoint["state_dict"].items()
                if k.startswith("model.")
            }

            scaler_weights = {
                k[7:]: v
                for k, v in checkpoint["state_dict"].items()
                if k.startswith("scaler.")
            }
    
            return arch_weights, scaler_weights

        arch_weights, scaler_weights = load_weights(embedding_weights)
        self.scaler = ChannelWiseScaler(num_params)
        self.scaler.load_state_dict(scaler_weights)
        self.embedding_net.load_state_dict(arch_weights)

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
