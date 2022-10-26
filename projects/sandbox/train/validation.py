from math import ceil
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from mlpe.data.transforms import WaveformInjector


def make_validation_dataset(
    background: np.ndarray,
    valid_injector: "WaveformInjector",
    kernel_length: float,
    stride: float,
    sample_rate: float,
    batch_size: int,
    device: str,
):

    kernel_size = int(kernel_length * sample_rate)
    stride_size = int(stride * sample_rate)
    num_kernels = (background.shape[-1] - kernel_size) // stride_size + 1
    num_kernels = int(num_kernels)
    num_ifos = len(background)

    waveforms, parameters = valid_injector.sample(-1)

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
    # unfold op orders things. Don't worry about it
    background = background.reshape(num_ifos, num_kernels, kernel_size)
    background = background.transpose(1, 0)

    # create repeats of background kernels
    # if we need any
    repeats = ceil(len(waveforms) / len(background))
    background = background.repeat(repeats, 1, 1)
    background = background[: len(waveforms)]

    # inject waveforms into center of kernel
    start = waveforms.shape[-1] // 2 - kernel_size // 2
    stop = start + kernel_size
    background += waveforms[:, :, start:stop]

    dataset = torch.utils.data.TensorDataset(background, parameters)
    return torch.utils.data.DataLoader(
        dataset,
        pin_memory=True,
        batch_size=batch_size,
        pin_memory_device=device,
    )
