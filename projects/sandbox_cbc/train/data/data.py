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
        self.device = device

        self.tensors, self.vertices = gw.get_ifo_geometry('H1', 'L1')

    def sample_waveforms(self, N: int):
        # sample parameters from prior
        parameters = self.prior.sample(N)
        # FIXME: generalize to other parameter combinations
        chirp_mass, mass_ratio = chirp_mass_mass_ratio(
            parameters["mass_1"], parameters["mass_2"]
        )
        parameters["chirp_mass"] = chirp_mass
        parameters["mass_ratio"] = mass_ratio
        # parameters = torch.from_numpy(parameters).to(device=self.device)
        # generate intrinsic waveform
        hf_p, hf_c = self.waveform_generator.frequency_domain_strain(
            torch.from_numpy(parameters["chirp_mass"]).to(device=self.device,dtype=torch.float32),
            torch.from_numpy(parameters["mass_ratio"]).to(device=self.device,dtype=torch.float32),
            torch.from_numpy(parameters["a_1"]).to(device=self.device,dtype=torch.float32),
            torch.from_numpy(parameters["a_2"]).to(device=self.device,dtype=torch.float32),
            torch.from_numpy(parameters["luminosity_distance"]).to(device=self.device,dtype=torch.float32),
            torch.from_numpy(parameters["phase"]).to(device=self.device,dtype=torch.float32),
            torch.from_numpy(parameters["theta_jn"]).to(device=self.device,dtype=torch.float32)
        )
        plus, cross = self.waveform_generator.time_domain_strain()

        waveforms = gw.compute_observed_strain(
            torch.from_numpy(parameters["dec"]).to(device=self.device,dtype=torch.float32),
            torch.from_numpy(parameters["psi"]).to(device=self.device,dtype=torch.float32),
            torch.from_numpy(parameters["ra"]).to(device=self.device,dtype=torch.float32),
            detector_tensors=self.tensors,
            detector_vertices=self.vertices,
            sample_rate=self.waveform_generator.sampling_frequency,
            plus=plus,
            cross=cross,
        )
        return parameters, waveforms

    def waveform_injector(self, X):
        N = len(X)
        kernel_size = X.shape[-1]
        parameters, waveforms = self.sample_waveforms(N)
        start = 0
        stop = kernel_size
        waveforms = waveforms[:, :, start:stop]
        X += waveforms
        return parameters, waveforms

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        X = super().__next__()
        parameters, X = self.waveform_injector(X)
        return parameters, X


class SignalDataSet(pl.LightningDataModule):
    def __init__(
        self,
        background_path: Path,
        ifos: Sequence[str],
        valid_frac: float,
        batch_size: int,
        batches_per_epoch: int,
        sampling_frequency: float,
        time_duration: float,
        f_min: float,
        f_max: float,
        f_ref: float,
        approximant=IMRPhenomD,
        prior: dict = nonspin_bbh(),
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.background_path = background_path
        self.num_ifos = len(ifos)
        self.prior = prior

        self.tensors, self.vertices = gw.get_ifo_geometry(*ifos)

    def load_background(self):
        background = []
        with h5py.File(self.background_path) as f:
            for ifo in self.hparams.ifos:
                hoft = f[ifo][:]
                background.append(hoft)
        return torch.from_numpy(np.stack(background)).to(dtype=torch.float32)

    def set_waveform_generator(self):
        self.waveform_generator = FrequencyDomainWaveformGenerator(
            self.hparams.time_duration,
            self.hparams.sampling_frequency,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.f_ref,
            self.hparams.approximant,
            device='cpu',
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
            self.hparams.time_duration,
            self.hparams.sampling_frequency,
            scaler=self.standard_scaler,
        )
        self.preprocessor.whitener.fit(
            self.hparams.time_duration,
            *self.background
        )
        # self.preprocessor.whitener.to(self.device)
        # set waveform generator and initialize in-memory datasets
        self.set_waveform_generator()
        self.training_dataset = PEInMemoryDataset(
            self.background,
            waveform_generator=self.waveform_generator,
            prior=self.prior,
            kernel_size=self.waveform_generator.number_of_samples,
            batch_size=self.hparams.batch_size,
            batches_per_epoch=self.hparams.batches_per_epoch,
            coincident=False,
            shuffle=True,
            device='cpu',
        )
        self.validation_dataset = PEInMemoryDataset(
            self.valid_background,
            waveform_generator=self.waveform_generator,
            prior=self.prior,
            kernel_size=self.waveform_generator.time_duration,
            batch_size=self.hparams.batch_size,
            batches_per_epoch=self.hparams.batches_per_epoch,
            coincident=False,
            shuffle=True,
            device='cpu',
        )

    def train_dataloader(self):
        #pin_memory = isinstance(
        #    self.trainer.accelerator, pl.accelerators.CUDAAccelerator
        #)
        pin_memory = True

        local_world_size = getattr(self.trainer, 'device_ids', 1)
        num_workers = max(2, int(os.cpu_count() / local_world_size))
        #dataloader = torch.utils.data.DataLoader(
        #    self.training_dataset,
        #    num_workers=num_workers,
        #    pin_memory=pin_memory,
        #)
        #return dataloader
        return self.training_dataset

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=False,
        )
