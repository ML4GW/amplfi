from typing import Union

import numpy as np
import torch

TENSORABLE = Union[float, np.ndarray, torch.Tensor]


def _make_tensor(
    value: TENSORABLE,
    device: torch.device,
    dtype: torch.Type = torch.float32,
):
    """
    Quick utility function for more robustly
    initializing scalar tensors
    """
    try:
        value = torch.tensor(value, dtype=dtype)
    except TypeError:
        value = torch.tensor([value], dtype=dtype)
    return value.to(device)


class Transform(torch.nn.Module):
    def fit(self, X: torch.Tensor):
        pass

    def add_parameter(
        self,
        data: TENSORABLE,
        device: torch.device = "cpu",
        dtype: torch.Type = torch.float32,
    ) -> torch.nn.Parameter:
        value = _make_tensor(data, device, dtype)
        return torch.nn.Parameter(value, requires_grad=False)

    def set_value(
        self, parameter: torch.nn.Parameter, data: TENSORABLE
    ) -> None:
        tensor = _make_tensor(data, parameter.device, parameter.dtype)
        if tensor.shape != parameter.shape:
            raise ValueError(
                "Tried to set parameter with shape {} "
                "to data with shape {}".format(parameter.shape, tensor.shape)
            )
        parameter.data = tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
