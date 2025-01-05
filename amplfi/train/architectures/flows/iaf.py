from typing import Literal

import torch
import torch.distributions as dist
from pyro.distributions.conditional import ConditionalComposeTransformModule
from pyro.distributions.transforms import (
    ConditionalAffineAutoregressive,
    ConditionalSplineAutoregressive,
)
from pyro.nn import ConditionalAutoRegressiveNN

from . import FlowArchitecture


def conditional_spline_autoregressive(
    input_dim,
    context_dim,
    activation=None,
    hidden_dims=None,
    count_bins=8,
    bound=3.0,
):
    if hidden_dims is None:
        hidden_dims = [input_dim * 10, input_dim * 10]

    param_dims = [count_bins, count_bins, count_bins]

    arn = ConditionalAutoRegressiveNN(
        input_dim,
        context_dim,
        hidden_dims,
        nonlinearity=activation,
        param_dims=param_dims,
    )
    return ConditionalSplineAutoregressive(
        input_dim, arn, count_bins=count_bins, bound=bound, order="quadratic"
    )


def conditional_affine_autoregressive(
    input_dim, context_dim, hidden_dims, activation=None
):
    arn = ConditionalAutoRegressiveNN(
        input_dim,
        context_dim,
        hidden_dims,
        nonlinearity=activation,
    )
    return ConditionalAffineAutoregressive(arn)


class InverseAutoregressiveFlow(FlowArchitecture):
    def __init__(
        self,
        *args,
        hidden_features: int = 50,
        num_transforms: int = 5,
        num_blocks: int = 2,
        activation: torch.nn.modules.activation = torch.nn.Tanh(),
        transform_type: Literal["spline", "affine"] = "spline",
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.transform_type = transform_type
        self.hidden_features = hidden_features
        self.num_blocks = num_blocks
        self.num_transforms = num_transforms
        self.activation = activation

        # register these as buffers so the
        # distributions are moved to the correct device
        self.register_buffer("mean", torch.zeros(self.num_params))
        self.register_buffer("std", torch.ones(self.num_params))

        # build the sequence of transforms
        self.transforms = self.build_transforms()

    def transform_block(self):
        if self.transform_type == "affine":
            return conditional_affine_autoregressive(
                self.num_params,
                self.embedding_net.context_dim,
                hidden_dims=[self.hidden_features] * self.num_blocks,
                activation=self.activation,
            )
        elif self.transform_type == "spline":
            return conditional_spline_autoregressive(
                self.num_params,
                self.embedding_net.context_dim,
                hidden_dims=[self.hidden_features] * self.num_blocks,
                activation=self.activation,
            )
        else:
            raise ValueError(
                f"Transform type {self.transform_type} not recognized. "
                "Must be one of ['spline', 'affine']"
            )

    def distribution(self):
        """Returns the base distribution for the flow"""
        return dist.Normal(
            self.mean,
            self.std,
        )

    def build_transforms(self):
        """Build the transform"""
        transforms = []
        for _ in range(self.num_transforms):
            transform = self.transform_block()
            transforms.extend([transform])
        return ConditionalComposeTransformModule(transforms)


class MaskedAutoregressiveFlow(InverseAutoregressiveFlow):
    """Affine autoregressive transforms that allow density
    evaluation in a single forward pass."""

    def transform_block(self):
        t = super().transform_block()
        return t.inv
