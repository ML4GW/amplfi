from math import pi
from typing import List, Optional, Tuple, TypeVar

import numpy as np
import torch

from ml4gw.distributions import Cosine, Uniform
from mlpe.data.transforms import WaveformInjector

Tensor = TypeVar("T", np.ndarray, torch.Tensor)

# TODO: how to generalize this to be able
# to pass arbitrary distributions as function argument
EXTRINSIC_DISTS = {
    # uniform on sky
    "dec": Cosine(),
    "psi": Uniform(0, pi),
    "phi": Uniform(-pi, pi),
}


def split(X: Tensor, frac: float, axis: int) -> Tuple[Tensor, Tensor]:
    """
    Split an array into two parts along the given axis
    by an amount specified by `frac`. Generic to both
    numpy arrays and torch Tensors.
    """

    size = int(frac * X.shape[axis])
    if isinstance(X, np.ndarray):
        return np.split(X, [size], axis=axis)
    else:
        splits = [size, X.shape[axis] - size]
        return torch.split(X, splits, dim=axis)


def prepare_augmentation(
    signals: np.ndarray,
    ifos: List[str],
    valid_frac: float,
    sample_rate: float,
    trigger_distance: float,
    highpass: float,
    intrinsic: Optional[np.ndarray] = None,
):

    valid_injector = None

    # construct validation injector
    # if valid_frac is passed
    if valid_frac is not None:

        signals, valid_signals = split(signals, 1 - valid_frac, 0)
        valid_plus, valid_cross = valid_signals.transpose(1, 0, 2)

        # if we've passed intrinsic parameters,
        # split them into validation and training
        if intrinsic is not None:
            intrinsic = intrinsic.transpose(1, 0)
            intrinsic, valid_intrinsic = split(intrinsic, 1 - valid_frac, 0)
        # otherwise, set valid_intrinsic to None
        # so that the injector doesnt return samples from it
        else:
            valid_intrinsic = None

        # TODO: make extrinisic parameter
        # sampling deterministic
        valid_injector = WaveformInjector(
            sample_rate,
            ifos,
            dec=EXTRINSIC_DISTS["dec"],
            psi=EXTRINSIC_DISTS["psi"],
            phi=EXTRINSIC_DISTS["phi"],
            intrinsic_parameters=valid_intrinsic,
            highpass=highpass,
            trigger_offset=trigger_distance,
            plus=valid_plus,
            cross=valid_cross,
        )

    plus, cross = signals.transpose(1, 0, 2)

    # prepare injector
    injector = WaveformInjector(
        sample_rate,
        ifos,
        dec=EXTRINSIC_DISTS["dec"],
        psi=EXTRINSIC_DISTS["psi"],
        phi=EXTRINSIC_DISTS["phi"],
        intrinsic_parameters=intrinsic,
        trigger_offset=trigger_distance,
        plus=plus,
        cross=cross,
    )

    return injector, valid_injector
