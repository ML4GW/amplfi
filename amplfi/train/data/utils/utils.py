from typing import Callable, Dict, Optional

import torch


class ParameterTransformer(torch.nn.Module):
    """
    Helper class for applying preprocessing
    transformations to inference parameters
    """

    def __init__(self, **transforms: Callable):
        super().__init__()
        self.transforms = transforms

    def forward(
        self,
        parameters: Dict[str, torch.Tensor],
    ):
        # transform parameters
        transformed = {k: v(parameters[k]) for k, v in self.transforms.items()}
        # update parameter dict
        parameters.update(transformed)
        return parameters


class ParameterSampler(torch.nn.Module):
    def __init__(
        self,
        conversion_function: Optional[Callable] = None,
        **parameters: Callable,
    ):
        """
        A class for sampling parameters from a prior distribution

        Args:
            conversion_function:
                A callable that takes a dictionary of sampled parameters
                and returns a dictionary of waveform generation parameters
            **parameters:
                A dictionary of parameter samplers that take an integer N
                and return a tensor of shape (N, ...) representing
                samples from the prior distribution
        """
        super().__init__()
        self.parameters = parameters
        self.conversion_function = conversion_function or (lambda x: x)

    def forward(
        self,
        N: int,
        device: str = "cpu",
    ):
        # sample parameters from prior
        parameters = {
            k: v.sample((N,)).to(device) for k, v in self.parameters.items()
        }
        # perform any necessary conversions
        # to from sampled parameters to
        # waveform generation parameters
        parameters = self.conversion_function(parameters)
        return parameters


class ZippedDataset(torch.utils.data.IterableDataset):
    def __init__(self, *datasets):
        super().__init__()
        self.datasets = datasets

    def __len__(self):
        lengths = []
        for dset in self.datasets:
            try:
                lengths.append(len(dset))
            except Exception as e:
                raise e from None
        return min(lengths)

    def __iter__(self):
        return zip(*self.datasets, strict=False)
