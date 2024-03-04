from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Optional, Tuple, TypeVar

import h5py
import lightning.pytorch as pl
import numpy as np
import torch
from waveforms import FrequencyDomainWaveformGenerator

from ml4gw import gw
from ml4gw.dataloading import InMemoryDataset
from ml4gw.transforms import ChannelWiseScaler
from ml4gw.waveforms import IMRPhenomD, TaylorF2
from mlpe.data.transforms import Preprocessor
from mlpe.injection.priors import nonspin_bbh_chirp_mass_q_parameter_sampler

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
        preprocessor: Optional[Callable] = None,
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
        self.preprocessor = preprocessor
        self.prior = prior
        self.device = device

        self.tensors, self.vertices = gw.get_ifo_geometry("H1", "L1")
        self.tensors = self.tensors.to(self.device)
        self.vertices = self.vertices.to(self.device)

    def sample_waveforms(self, N: int):
        # sample parameters from prior
        parameters = self.prior(N)
        intrinsic_parameters = torch.vstack(
            (
                parameters["chirp_mass"],
                parameters["mass_ratio"],
                #parameters["mass_1"],
                #parameters["mass_2"],
                parameters["a_1"],
                parameters["a_2"],
                parameters["luminosity_distance"],
                parameters["phase"],
                parameters["theta_jn"],
            )
        )
        # FIXME: generalize to other parameter combinations
        # generate intrinsic waveform
        plus, cross = self.waveform_generator.time_domain_strain(
            *intrinsic_parameters
        )
        dec_psi_ra = torch.vstack(
            (
                parameters["dec"],
                parameters["psi"],
                parameters["phi"]
            )
        )

        waveforms = gw.compute_observed_strain(
            *dec_psi_ra,
            detector_tensors=self.tensors,
            detector_vertices=self.vertices,
            sample_rate=self.waveform_generator.sampling_frequency,
            plus=plus,
            cross=cross,
        )
        # FIXME: delta function distributions are removed, make it cleaner
        intrinsic_parameters = torch.vstack(
            (intrinsic_parameters[:2], intrinsic_parameters[4:]))
        return torch.vstack((intrinsic_parameters, dec_psi_ra)), waveforms

    def waveform_injector(self, X):
        N = len(X)
        kernel_size = X.shape[-1]
        parameters, waveforms = self.sample_waveforms(N)
        start = 0
        stop = kernel_size
        waveforms = waveforms[:, :, start:stop]
        X += waveforms
        return parameters, X, waveforms

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        X = super().__next__()
        parameters, X, waveforms = self.waveform_injector(X)
        # whiten and scale parameters
        if self.preprocessor:
            transformed_X, transformed_parameters = self.preprocessor(
                X, parameters.T
            )
        return transformed_X.to(dtype=torch.float32), transformed_parameters.to(dtype=torch.float32)


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
        approximant=TaylorF2,
        prior_func: callable = nonspin_bbh_chirp_mass_q_parameter_sampler,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.background_path = background_path
        self.num_ifos = len(ifos)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prior_func = prior_func  # instantiate in setup

        self.tensors, self.vertices = gw.get_ifo_geometry(*ifos)

    def load_background(self):
        background = []
        with h5py.File(self.background_path) as f:
            for ifo in self.hparams.ifos:
                hoft = f[ifo][:]
                background.append(hoft)
        return torch.from_numpy(np.stack(background)).to(dtype=torch.float64)

    def set_waveform_generator(self):
        self.waveform_generator = FrequencyDomainWaveformGenerator(
            self.hparams.time_duration,
            self.hparams.sampling_frequency,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.f_ref,
            self.hparams.approximant,
            start_time=-0.5,
            post_padding=1.0,
            device=self.device,
        )

    def setup(self, stage: str) -> None:
        # load background and fit whitener
        background = self.load_background()
        self.background, self.valid_background = split(
            background, 1 - self.hparams.valid_frac, 1
        )
        self.valid_background, self.test_background = split(
            self.valid_background, 0.5, 1
        )
        self.standard_scaler = ChannelWiseScaler(8)  # FIXME: don't hardcode
        # self.standard_scaler = torch.nn.Identity()
        # FIXME: clean up the standard scaler fitting
        self.prior = self.prior_func(self.device)
        _samples = self.prior(10000)
        _samples = torch.vstack((
            _samples["chirp_mass"],
            _samples["mass_ratio"],
            #_samples["mass_1"],
            #_samples["mass_2"],
            _samples["luminosity_distance"],
            _samples["phase"],
            _samples["theta_jn"],
            _samples["dec"],
            _samples["psi"],
            _samples["phi"]))
        self.standard_scaler.fit(_samples)
        self.standard_scaler.to(self.device)
        self.preprocessor = Preprocessor(
            self.num_ifos,
            self.hparams.time_duration + 1,
            self.hparams.sampling_frequency,
            scaler=self.standard_scaler,
        )
        self.preprocessor.whitener.fit(1, *background, fftlength=2)
        self.preprocessor.whitener.to(self.device)
        # set waveform generator and initialize in-memory datasets
        self.set_waveform_generator()
        # FIXME: check if this is OK
        self.training_dataset = PEInMemoryDataset(
            self.background,
            waveform_generator=self.waveform_generator,
            prior=self.prior,
            preprocessor=self.preprocessor,
            kernel_size=self.waveform_generator.number_of_samples
            + self.waveform_generator.number_of_post_padding,
            batch_size=self.hparams.batch_size,
            batches_per_epoch=self.hparams.batches_per_epoch,
            coincident=False,
            shuffle=True,
            device=self.device,
        )
        self.validation_dataset = PEInMemoryDataset(
            self.valid_background,
            waveform_generator=self.waveform_generator,
            prior=self.prior,
            preprocessor=self.preprocessor,
            kernel_size=self.waveform_generator.number_of_samples
            + self.waveform_generator.number_of_post_padding,
            batch_size=self.hparams.batch_size,
            batches_per_epoch=self.hparams.batches_per_epoch,
            coincident=False,
            shuffle=False,
            device=self.device,
        )
        self.test_dataset = PEInMemoryDataset(
            self.test_background,
            waveform_generator=self.waveform_generator,
            prior=self.prior,
            preprocessor=self.preprocessor,
            kernel_size=self.waveform_generator.number_of_samples
            + self.waveform_generator.number_of_post_padding,
            batch_size=1,
            batches_per_epoch=self.hparams.batches_per_epoch,
            coincident=False,
            shuffle=False,
            device=self.device,
        )

    def train_dataloader(self):
        return self.training_dataset

    def val_dataloader(self):
        return self.validation_dataset

    def test_dataloader(self):
        return self.test_dataset
