from pathlib import Path

import h5py
import torch

from .sampler import WaveformSampler


def x_per_y(x, y):
    return int((x - 1) // y) + 1


class WaveformLoader(WaveformSampler):
    """
    Torch module for loading waveforms from disk,
    performing train/val/test split, and sampling
    them during training.

    TODO: modify this to sample waveforms from disk, taking
    an index sampler object so that DDP training can sample
    different waveforms for each device.

    Args:
        waveform_file:
            Path to the HDF5 file containing the waveforms
        val_frac:
            Fraction of waveforms to use for validation
    """

    def __init__(
        self,
        *args,
        waveform_file: Path,
        val_frac: float,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)
        self.val_frac = val_frac
        self.waveform_file = waveform_file

        with h5py.File(waveform_file) as f:
            self.num_waveforms = len(f["signals"])

        self.waveform_file = waveform_file
        (
            self.train_waveforms,
            self.train_parameters,
        ) = self.get_train_waveforms()

    @property
    def num_val_waveforms(self):
        """Total number of validation waveforms across all devices"""
        return int(self.val_frac * self.num_waveforms)

    @property
    def val_waveforms_per_device(self):
        """Number of validation waveforms per device"""
        world_size, _ = self.get_world_size_and_rank()
        return self.num_val_waveforms // world_size

    @property
    def num_train_waveforms(self):
        """Total number of training waveforms"""
        return self.num_waveforms - self.num_val_waveforms

    def load_signals(self, start, stop):
        """
        Load signals and parameters of specified indices from the dataset
        """
        with h5py.File(self.waveform_file) as f:
            signals = torch.Tensor(f["signals"][start:stop])
            parameters = {}
            for parameter in self.inference_params:
                parameters[parameter] = torch.Tensor(f[parameter][start:stop])

        return signals, parameters

    def get_slice_bounds(self, total, world_size, rank) -> tuple[int, int]:
        """
        Determine waveform indices to load for this device
        given our rank and world size
        """
        per_dev = x_per_y(abs(total), world_size)
        start = rank * per_dev
        stop = (rank + 1) * per_dev
        return start, stop

    def get_train_waveforms(
        self,
    ):
        """
        Returns train waveforms for this device
        """
        world_size, rank = self.get_world_size_and_rank()
        start, stop = self.get_slice_bounds(
            self.num_train_waveforms, world_size, rank
        )
        return self.load_signals(start, stop)

    def get_val_waveforms(self):
        """
        Returns validation waveforms for this device
        """
        world_size, rank = self.get_world_size_and_rank()
        start, stop = self.get_slice_bounds(
            self.num_val_waveforms, world_size, rank
        )
        # start counting from the back for val waveforms
        start, stop = -start, -stop or None
        return self.load_signals(start, stop)

    def get_test_waveforms(self, f, world_size, rank):
        """
        Load test waveforms
        """
        return

    def slice_waveforms(self, waveforms: torch.Tensor):
        """
        Slice waveforms to the desired length;
        **NOTE** it is assumed here that waveforms are centered;
        """
        center = waveforms.shape[-1] // 2
        half = self.waveform_length // 2
        start, stop = center - half, center + half
        return waveforms[:, start:stop]

    def sample(self, X):
        """
        Sample method for generating training waveforms
        """
        N = X.shape[0]

        idx = torch.randperm(len(self.train_waveforms))[:N]
        waveforms = self.train_waveforms[idx]
        parameters = {}
        for k, v in self.train_parameters.items():
            parameters[k] = v[idx]

        cross, plus = waveforms
        polarizations = {"cross": cross, "plus": plus}

        return polarizations, parameters
