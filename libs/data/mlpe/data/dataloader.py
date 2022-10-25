from typing import Callable, Optional, Tuple

import numpy as np
import torch

from ml4gw.dataloading import InMemoryDataset


class PEInMemoryDataset(InMemoryDataset):
    """
    Dataloader which samples batches of kernels
    from a single timeseries array and prepares
    corresponding target array of all 0s. Optionally
    applies a preprocessing step to both the sampled
    kernels and their targets.
    Args:
        X: Array containing multi-channel timeseries data
        kernel_size:
            The size of the kernels, in terms of number of
            samples, to sample from the timeseries.
        batch_size:
            Number of kernels to produce at each iteration.
            Represents the 0th dimension of the returned tensor.
        batches_per_epoch:
            Number of iterations dataset will perform before
            raising a `StopIteration` exception.
        preprocessor:
            Optional preprocessing step to apply to both the
            sampled kernels and their targets. If left as
            `None`, the batches and targets will be returned
            as-is.
        coincident:
            Whether to sample kernels from all channels using
            the same timesteps, or whether to sample them
            independently from across the whole timeseries.
        shuffle:
            Whether to samples kernels uniformly from the
            timeseries, or iterate through them in order.
        device:
            Device on which to host the timeseries dataset.
    """

    def __init__(
        self,
        X: np.ndarray,
        kernel_size: int,
        batch_size: int = 32,
        preprocessor: Optional[Callable] = None,
        batches_per_epoch: Optional[int] = None,
        coincident: bool = True,
        shuffle: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            X,
            kernel_size,
            batch_size=batch_size,
            stride=1,
            batches_per_epoch=batches_per_epoch,
            coincident=coincident,
            shuffle=shuffle,
            device=device,
        )
        self.preprocessor = preprocessor

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        X = super().__next__()
        if self.preprocessor is not None:
            X, parameters = self.preprocessor(X)
            return X, parameters

        return X
