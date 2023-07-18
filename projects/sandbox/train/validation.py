from math import ceil
from typing import List

import numpy as np
import torch

from ml4gw import gw


def make_validation_dataset(
    background: np.ndarray,
    signals: np.ndarray,
    parameters: np.ndarray,
    ifos: List[str],
    kernel_length: float,
    stride: float,
    sample_rate: float,
    batch_size: int,
    device: str,
):

    tensors, vertices = gw.get_ifo_geometry(*ifos)
    dec, psi, phi = parameters[:, -3:].transpose(1, 0)

    cross, plus = signals.transpose(1, 0, 2)
    signals = gw.compute_observed_strain(
        torch.Tensor(dec),
        torch.Tensor(psi),
        torch.Tensor(phi),
        detector_tensors=tensors,
        detector_vertices=vertices,
        sample_rate=sample_rate,
        plus=torch.Tensor(plus),
        cross=torch.Tensor(cross),
    )

    kernel_size = int(kernel_length * sample_rate)
    center = signals.shape[-1] // 2
    start = center - (kernel_size // 2)
    stop = center + (kernel_size // 2)
    signals = signals[:, :, start:stop]

    stride_size = int(stride * sample_rate)
    num_kernels = (background.shape[-1] - kernel_size) // stride_size + 1
    num_kernels = int(num_kernels)
    num_ifos = len(background)

    # Create pre-computed kernels of pure background
    # slice our background so that it has an integer number of
    # windows, then add dummy dimensions since unfolding only
    # works on 4D tensors
    background = background[:, : num_kernels * stride_size + kernel_size]
    background = torch.Tensor(background).view(1, num_ifos, 1, -1)

    # fold out into windows up front
    background = torch.nn.functional.unfold(
        background, (1, num_kernels), dilation=(1, stride_size)
    )

    # some reshape magic having to do with how the
    # unfold op orders things
    background = background.reshape(num_ifos, num_kernels, kernel_size)
    background = background.transpose(1, 0)

    # create repeats of background kernels if we need any
    repeats = ceil(len(signals) / len(background))
    background = background.repeat(repeats, 1, 1)
    background = background[: len(signals)]
    background += signals

    parameters = torch.Tensor(parameters)
    dataset = torch.utils.data.TensorDataset(background, parameters)
    return torch.utils.data.DataLoader(
        dataset,
        pin_memory=True,
        batch_size=batch_size,
        pin_memory_device=device,
    )
