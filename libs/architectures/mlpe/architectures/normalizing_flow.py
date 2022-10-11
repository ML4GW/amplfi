from typing import Callable, Optional

import nflows.nn.nets as nn_
import torch
from nflows import distributions, flows, transforms, utils


def create_flow(
    param_dim: int,
    context_dim: int,
    num_flow_steps: int,
    embedding_net: Optional[torch.nn.Module] = None,
    **base_transform_kwargs
):

    """
    Creates a normalizing flow with nflows that
    uses a piecewise rational quadratic coupling transform.

    Args:
        param_dim:
            Number of parameters on
            which inference is being performed
        context_dim:
            Dimension of the context on which
            the parameters are conditioned
        num_flow_steps:
            Number of flow steps
        embedding_net:
            A torch.nn.Module that will embed the context
            in a lower dimensional space before conditioning. Trained
            alongside the flow
        **base_transform_kwargs:
            kwargs passed to the base transform
    """

    # base distribution to learn transform for
    distribution = distributions.StandardNormal((param_dim,))

    transform = transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    create_linear_transform(param_dim),
                    create_base_transform(
                        i,
                        param_dim,
                        context_dim,
                        **base_transform_kwargs,
                    ),
                ]
            )
            for i in range(num_flow_steps)
        ]
        + [create_linear_transform(param_dim)]
    )

    # create flow transform
    flow = flows.Flow(transform, distribution, embedding_net=embedding_net)

    flow.model_hyperparams = {
        "param_dim": param_dim,
        "num_flow_steps": num_flow_steps,
        "context_dim": context_dim,
    }

    return flow


def create_base_transform(
    idx: int,
    param_dim: int,
    context_dim: int,
    hidden_dim: int = 512,
    num_transform_blocks: int = 2,
    activation: Callable = torch.nn.functional.relu,
    dropout_probability: float = 0.0,
    batch_norm: bool = False,
    num_bins: int = 8,
    tails: str = "linear",
    tail_bound: float = 1.0,
    apply_unconditional_transform: bool = False,
):
    transform = transforms.PiecewiseRationalQuadraticCouplingTransform(
        mask=utils.create_alternating_binary_mask(
            param_dim, even=(idx % 2 == 0)
        ),
        transform_net_create_fn=(
            lambda in_features, out_features: nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_dim,
                context_features=context_dim,
                num_blocks=num_transform_blocks,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm,
            )
        ),
        num_bins=num_bins,
        tails=tails,
        tail_bound=tail_bound,
        apply_unconditional_transform=apply_unconditional_transform,
    )

    return transform


def create_linear_transform(param_dim: int):

    return transforms.CompositeTransform(
        [
            transforms.RandomPermutation(features=param_dim),
            transforms.LULinear(param_dim, identity_init=True),
        ]
    )
