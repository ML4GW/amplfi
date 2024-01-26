import os
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Tuple, TypeVar

import h5py
import lightning.pytorch as pl
import numpy as np
import torch
from waveforms import FrequencyDomainWaveformGenerator

from ml4gw import gw
from ml4gw.dataloading import InMemoryDataset
from ml4gw.transforms import ChannelWiseScaler
from ml4gw.waveforms import IMRPhenomD
from mlpe.data.transforms import Preprocessor
from mlpe.injection.priors import nonspin_bbh


def chirp_mass_mass_ratio(m1, m2):
    chirp_mass = (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)
    mass_ratio = m2 / m1
    return chirp_mass, mass_ratio


def x_per_y(x, y):
    return int((x - 1) // y) + 1


Tensor = TypeVar("T", np.ndarray, torch.Tensor)


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


class PEInMemoryDataset(InMemoryDataset):
    def __init__(
        self,
        X: np.ndarray,
        waveform_generator: FrequencyDomainWaveformGenerator,
        prior: dict,
        kernel_size: int,
        batch_size: int = 32,
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
        self.waveform_generator = waveform_generator
        self.prior = prior

    def sample_waveforms(self, N: int):
        # sample parameters from prior
        parameters = self.prior.sample(N)
        parameters = torch.from_numpy(parameters).to(device=self.device)
        # FIXME: generalize to other parameter combinations
        chirp_mass, mass_ratio = chirp_mass_mass_ratio(
            parameters["mass_1"], parameters["mass_2"]
        )
        parameters["chirp_mass"] = chirp_mass
        parameters["mass_ratio"] = mass_ratio
        # generate intrinsic waveform
        hf_p, hf_c = self.waveform_generator.frequency_domain_strain(
            parameters["chirp_mass"],
            parameters["mass_ratio"],
            parameters["a_1"],
            parameters["a_2"],
            parameters["luminosity_distance"],
            parameters["phase"],
            parameters["theta_jn"],
        )
        plus, cross = self.waveform_generator.time_domain_strain()

        waveforms = gw.compute_observed_strain(
            parameters["dec"],
            parameters["psi"],
            parameters["ra"],
            detector_tensors=self.tensors,
            detector_vertices=self.vertices,
            sample_rate=self.sample_rate,
            plus=plus,
            cross=cross,
        )
        return parameters, waveforms

    def waveform_injector(self, X):
        N = len(X)
        kernel_size = X.shape[-1]
        waveforms, parameters = self.sample_waveforms(N)
        start = 0
        stop = kernel_size
        waveforms = waveforms[:, :, start:stop]
        X += waveforms
        return X, parameters

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        X = super().__next__()
        X, parameters = self.waveform_injector(X)
        return X


class SignalDataSet(pl.LightningDataModule):
    def __init__(
        self,
        background_path: Path,
        ifos: Sequence[str],
        valid_frac: float,
        batch_size: int,
        sampling_frequency: float,
        time_duration: float,
        f_min: float,
        f_max: float,
        approximant=IMRPhenomD,
        prior: dict = nonspin_bbh,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.background_path = background_path
        self.num_ifos = len(ifos)
        self.prior = prior

        tensors, vertices = gw.get_ifo_geometry(*ifos)
        self.register_buffer("tensors", tensors)
        self.register_buffer("vertices", vertices)

    def load_background(self):
        background = []
        with h5py.File(self.background_path) as f:
            for ifo in self.ifos:
                hoft = f[ifo][:]
                background.append(hoft)
        return np.stack(background)

    def set_waveform_generator(self):
        self.waveform_generator = FrequencyDomainWaveformGenerator(
            self.hparams.time_duration,
            self.hparams.sampling_frequency,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.approximant,
            device=self.device,
        )

    def setup(self, stage: str) -> None:
        # load background and fit whitener
        background = self.load_background()
        self.background, self.valid_background = split(
            background, 1 - self.hparams.valid_frac, 1
        )
        self.standard_scaler = ChannelWiseScaler(len(self.prior))
        self.preprocessor = Preprocessor(
            self.num_ifos,
            self.hparams.sampling_frequency,
            self.hparams.time_duration,
            scaler=self.standard_scaler,
        )
        self.preprocessor.whitener.fit(
            kernel_length=2,
            highpass=self.f_min,
            sample_rate=self.hparams.sampling_frequency,
            *self.background
        )
        self.preprocessor.whitener.to(self.device)
        # set waveform generator and initialize in-memory datasets
        self.set_waveform_generator()
        self.training_dataset = PEInMemoryDataset(
            self.background,
            kernel_size=self.waveform_generator.number_of_samples,
            batch_size=self.hparams.batch_size,
            stride=1,
            batches_per_epoch=self.hparams.batches_per_epoch,
            coincident=False,
            shuffle=True,
            device=self.device,
        )
        self.validation_dataset = PEInMemoryDataset(
            self.valid_background,
            kernel_size=self.waveform_generator.number_of_samples,
            batch_size=self.hparams.batch_size,
            stride=1,
            batches_per_epoch=self.hparams.batches_per_epoch,
            coincident=False,
            shuffle=True,
            device=self.device,
        )

    def train_dataloader(self):
        pin_memory = isinstance(
            self.trainer.accelerator, pl.accelerators.CUDAAccelerator
        )

        local_world_size = len(self.trainer.device_ids)
        num_workers = min(6, int(os.cpu_count() / local_world_size))
        dataloader = torch.utils.data.DataLoader(
            self.training_dataset,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return dataloader

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=False,
        )
