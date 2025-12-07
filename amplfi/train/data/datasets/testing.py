"""
Additional Lightning DataModules for testing
amplfi models across various usecases
"""

from pathlib import Path
from typing import Optional, Union
import pandas as pd
import h5py
import numpy as np
import torch
from astropy.time import Time
from ml4gw.waveforms.conversion import chirp_mass_and_mass_ratio_to_components
from amplfi.train.data.datasets import FlowDataset
from ml4gw import gw
from ..utils.utils import ZippedDataset


def phi_from_ra(ra: np.ndarray, gpstimes: np.ndarray) -> float:
    # get the sidereal time at the observation time
    gmsts = []
    for t in gpstimes:
        t = Time(t, format="gps", scale="utc")
        gmst = t.sidereal_time("mean", "greenwich").to("rad").value
        gmsts.append(gmst)

    gmsts = np.array(gmsts)
    # calculate the relative azimuthal angle in the range [0, 2pi]
    phi = np.remainder(ra - gmsts, 2 * np.pi)

    return phi


def ra_from_phi(phi: torch.Tensor, gpstime: float):
    t = Time(gpstime, format="gps", scale="utc")
    gmst = t.sidereal_time("mean", "greenwich").to("rad").value
    ra = phi + gmst
    ra = torch.remainder(ra, 2 * np.pi)
    return ra


class StrainTestingDataset(FlowDataset):
    """
    Testing dataset for analyzing pre-made injections.

    Subclass of `amplfi.train.data.datasets.flow` and
    thus `amplfi.train.data.datasets.base`. See those classes
    for additional arguments and keyword arguments

    Args:
        dataset_path:
            Path to hdf5 file containing premade injections.
            For each interferometer being analyzed, the strain
            data should be stored in an hdf5 group named `strain` with
            each dataset within named after the interferometer.
            The dataset should be of shape (batch, time).
            It is assumed that the coalescence
            time of the injection is placed in the middle of each sample
            of the array. If this is not the case, supply appropriate offset
            using `middle_offset` keyword argument in seconds that is added
            to the middle to reach the coalescence time. In addition,
            each inference parameter should be saved in a group name
            `parameters` using pandas
            `dataframe.to_hdf(path, key='parameters')`.

    """

    def __init__(
        self, dataset_path: Path, *args, middle_offset: float = 0.0, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dataset_path = dataset_path
        self.i = 0
        self.middle_offset = middle_offset

    def setup(self, stage):
        world_size, rank = self.get_world_size_and_rank()
        self._logger = self.get_logger(world_size, rank)
        if stage != "test":
            raise ValueError(
                "StrainTestingDataset should only be used for testing"
            )

        parameters = pd.read_hdf(self.dataset_path, key="parameters")

        keys = self.hparams.inference_params.copy()
        keys.remove("phi")
        keys += ["ra", "gpstime"]
        parameters = parameters[keys]
        parameters["phi"] = phi_from_ra(
            parameters["ra"], parameters["gpstime"]
        )

        # load in the strain data and parameters
        strain = []

        with h5py.File(self.dataset_path, "r") as f:
            for ifo in self.hparams.ifos:
                strain.append(np.array(f["strain"][ifo]))
        strain = np.stack(strain, axis=1)
        strain = torch.tensor(strain, dtype=torch.float32)

        # based on psd length, fduration and kernel length, and padding,
        # determine slice indices. It is assumed the coalescence
        # time of the waveform is in the middle unless `middle_offset` is set
        middle_offset = int(self.middle_offset * self.hparams.sample_rate)
        middle = strain.shape[-1] // 2 + middle_offset
        post = self.waveform_sampler.right_pad + self.hparams.fduration / 2
        pre = (
            post
            - self.hparams.kernel_length
            - (self.hparams.fduration / 2)
            - self.hparams.psd_length
        )

        post = int(post * self.hparams.sample_rate)
        pre = int(pre * self.hparams.sample_rate)

        start, stop = middle + pre, middle + post
        strain = strain[..., start:stop]

        # the whole set of parameters used to generate the waveforms
        self.test_parameters: dict[str, torch.tensor] = parameters

        # torch tensor of just requested inference parameters
        # that the model is trained to draw samples for
        parameters = parameters[self.hparams.inference_params]
        parameters = torch.tensor(parameters.values)
        self.test_inference_params = parameters
        self.test_strain = strain

        # once we've generated validation/testing waveforms on cpu,
        # build data augmentation modules
        # and transfer them to appropiate device
        self.build_transforms(stage)
        self.transforms_to_device()

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        # build dataset and dataloader that will
        # simply load one injection (and its parameters) at a time
        dataset = torch.utils.data.TensorDataset(
            self.test_strain, self.test_inference_params
        )

        return torch.utils.data.DataLoader(
            dataset, batch_size=1, num_workers=12, shuffle=False
        )

    def on_after_batch_transfer(self, batch, _):
        """
        Override of the on_after_batch_transfer hook
        defined in `BaseDataset`.

        This is necessary since we're we don't want to
        do any injections (they are already in the data)
        """

        if not self.trainer.testing:
            raise ValueError(
                "Use of the StrainTestingDataset is for testing only"
            )

        X, parameters = batch

        X, psds = self.psd_estimator(X)
        X = self.whitener(X, psds)

        # scale parameters
        parameters = self.scale(parameters)

        # calculate asds, interpolating to size of
        # frequency array so we can concatenate them
        # into a single tensor for the embedding
        freqs = torch.fft.rfftfreq(X.shape[-1], d=1 / self.hparams.sample_rate)
        num_freqs = len(freqs)
        psds = torch.nn.functional.interpolate(
            psds, size=(num_freqs,), mode="linear"
        )

        mask = freqs > self.hparams.highpass
        psds = psds[:, :, mask]
        asds = torch.sqrt(psds)

        # return None for snrs, since for strain testing set
        # we don't have the raw waveforms to calculate SNRs
        return X, asds, parameters, None


class ParameterTestingDataset(FlowDataset):
    """
    Testing dataset for using `amplfi` on the fly generationg
    to create injections from a fixed, premade set of parameters

    Subclass of `amplfi.train.data.datasets.flow` and
    thus `amplfi.train.data.datasets.base`. See those classes
    for additional arguments and keyword arguments.

    Args:
        dataset_path:
            Path to hdf5 dataset containing parameters
            to inject. Should contain hdf5 datasets
            for each of the waveform generation parameters
        waveform_generation_parameters:
            An optional list of parameters that is used for generating
            waveforms. If left as `None`, the waveform generation
            parameters will be assumed to be the keys from the
            `waveform_sampler.training_prior` object.
    """

    def __init__(
        self,
        dataset_path: Path,
        *args,
        waveform_generation_parameters: Optional[list[str]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataset_path = dataset_path

        # by default, assume waveform generation parameters are those
        # specified in the parameter sampler parameter keys;
        # otherwise, use user passed parameters
        prior_keys = list(self.waveform_sampler.training_prior.priors.keys())

        self.waveform_generation_parameters = (
            prior_keys
            if waveform_generation_parameters is None
            else waveform_generation_parameters
        )

        # only apply parameter sampler conversion function
        # if the waveform_generation_parameters were not specified
        self.convert = waveform_generation_parameters is None
        self.i = 0

    def setup(self, stage: str):
        world_size, rank = self.get_world_size_and_rank()
        self._logger = self.get_logger(world_size, rank)
        if stage != "test":
            raise ValueError(
                "ParameterTestingDataset should only be used for testing"
            )

        parameters = pd.read_hdf(self.dataset_path, key="parameters")
        index = parameters.index

        load_keys = self.hparams.inference_params + [
            "ra",
            "dec",
            "psi",
            "gpstime",
        ]
        load_keys.remove("phi")
        if self.waveform_generation_parameters is not None:
            load_keys += self.waveform_generation_parameters

        parameters = parameters[load_keys]
        parameters = parameters.to_dict(orient="list")
        parameters = {k: torch.tensor(v) for k, v in parameters.items()}
        # apply conversion function to parameters
        # if we're generating from
        if self.convert:
            parameters = (
                self.waveform_sampler.training_prior.conversion_function(
                    parameters
                )
            )

        (
            parameters["mass_1"],
            parameters["mass_2"],
        ) = chirp_mass_and_mass_ratio_to_components(
            parameters["chirp_mass"], parameters["mass_ratio"]
        )
        # convert ra to phi
        parameters["phi"] = phi_from_ra(
            parameters["ra"], parameters["gpstime"]
        )
        # generate cross and plus using our infrastructure

        cross, plus = self.waveform_sampler(**parameters)

        # convert back to tensor
        params = []
        for k in self.hparams.inference_params:
            if k in parameters.keys():
                params.append(parameters[k])

        # torch tensor of inference parameters
        # that the model will draw samples for
        self.test_inference_params = torch.column_stack(params)

        # the whole set of parameters used to generate the waveforms
        self.test_parameters = pd.DataFrame(parameters, index=index)
        self.test_waveforms = torch.stack([cross, plus], dim=0)

        self.background, _ = self.background_from_gpstimes(
            parameters["gpstime"], self.get_test_fnames()
        )

        # once we've generated validation/testing waveforms on cpu,
        # build data augmentation modules
        # and transfer them to appropiate device
        self.build_transforms(stage)
        self.transforms_to_device()

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        cross, plus = self.test_waveforms

        waveform_dataset = torch.utils.data.TensorDataset(
            cross, plus, self.test_inference_params
        )

        waveform_dataloader = torch.utils.data.DataLoader(
            waveform_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=10,
        )

        background_dataset = torch.utils.data.TensorDataset(self.background)

        background_dataloader = torch.utils.data.DataLoader(
            background_dataset, pin_memory=False, num_workers=10, shuffle=False
        )
        return ZippedDataset(
            waveform_dataloader,
            background_dataloader,
        )

    def on_after_batch_transfer(self, batch, _):
        """
        Override of the on_after_batch_transfer hook
        defined in `BaseDataset`.
        """

        if not self.trainer.testing:
            raise ValueError(
                "Use of ParameterTestingDataset is for testing only"
            )

        [cross, plus, parameters], [X] = batch

        parameters = {
            k: parameters[:, i]
            for i, k in enumerate(self.hparams.inference_params)
        }

        dec, psi, phi = (
            parameters["dec"].float(),
            parameters["psi"].float(),
            parameters["phi"].float(),
        )
        waveforms = self.projector(dec, psi, phi, cross=cross, plus=plus)

        # make any requested parameter transforms
        parameters = self.transform(parameters)
        parameters = [
            torch.Tensor(parameters[k]) for k in self.hparams.inference_params
        ]
        parameters = torch.vstack(parameters).T

        X, psds = self.psd_estimator(X)

        num_freqs = waveforms.shape[-1] // 2 + 1
        psds = torch.nn.functional.interpolate(
            psds, size=(num_freqs,), mode="linear"
        )
        snrs = gw.compute_network_snr(
            waveforms,
            psds,
            self.hparams.sample_rate,
            self.hparams.highpass,
        )

        X += waveforms
        X = self.whitener(X, psds)

        # scale parameters
        parameters = self.scale(parameters)

        # calculate asds and highpass
        freqs = torch.fft.rfftfreq(X.shape[-1], d=1 / self.hparams.sample_rate)
        num_freqs = len(freqs)
        psds = torch.nn.functional.interpolate(
            psds, size=(num_freqs,), mode="linear"
        )

        mask = freqs > self.hparams.highpass
        psds = psds[:, :, mask]
        asds = torch.sqrt(psds)

        return X, asds, parameters, snrs


class RawStrainTestingDataset(FlowDataset):
    """
    Testing dataset for analyzing raw strain without injections
    from a given set of gpstimes.

    Useful for testing on real events. This dataset should only be
    used with the `predict` stage.

    Subclass of `amplfi.train.data.datasets.FlowDataset` and
    thus `amplfi.train.data.datasets.BaseDataset`. See those classes
    for additional arguments and keyword arguments.

    Args:
        gpstimes:
            A float, numpy array or path to an hdf5 file containing
            the gps times of the events to analyze. If a path is
            passed, the gps times should be stored in a dataset
            named `gpstimes`
    """

    def __init__(
        self, *args, gpstimes: Union[float, list[float], Path], **kwargs
    ):
        self.gpstimes = self.parse_gps_times(gpstimes)
        super().__init__(*args, **kwargs)

    def parse_gps_times(self, gpstimes: Union[float, np.ndarray, Path]):
        if isinstance(gpstimes, (float, int)):
            gpstimes = np.array([gpstimes])
        elif isinstance(gpstimes, list):
            gpstimes = np.array(gpstimes)
        elif isinstance(gpstimes, Path):
            with h5py.File(gpstimes, "r") as f:
                gpstimes = f["gpstimes"][:]

        return gpstimes

    def setup(self, stage: str):
        world_size, rank = self.get_world_size_and_rank()
        self._logger = self.get_logger(world_size, rank)
        if stage != "predict":
            raise ValueError(
                "RawStrainTestingDataset should only be used "
                "with `predict` stage"
            )

        self.background, gpstimes = self.background_from_gpstimes(
            self.gpstimes, self.get_test_fnames(), use_random_segment=False
        )
        self.gpstimes = gpstimes

        # once we've generated validation/testing waveforms on cpu,
        # build data augmentation modules
        # and transfer them to appropiate device
        self.build_transforms(stage)
        self.transforms_to_device()

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        dataset = torch.utils.data.TensorDataset(
            self.background, torch.tensor(self.gpstimes, dtype=torch.float64)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            pin_memory=False,
            num_workers=10,
            shuffle=False,
            batch_size=1,
        )
        return dataloader

    def on_after_batch_transfer(self, batch, _):
        """
        Override of the on_after_batch_transfer hook
        defined in `BaseDataset`.

        When we're analyzing
        real events, we don't need to do any injections
        """

        [X], gpstimes = batch

        X, psds = self.psd_estimator(X)
        X = self.whitener(X, psds)
        psds = psds[None]

        # calculate asds and highpass
        freqs = torch.fft.rfftfreq(X.shape[-1], d=1 / self.hparams.sample_rate)
        num_freqs = len(freqs)
        psds = torch.nn.functional.interpolate(
            psds, size=(num_freqs,), mode="linear"
        )

        mask = freqs > self.hparams.highpass
        psds = psds[:, :, mask]
        asds = torch.sqrt(psds)

        return X, asds, gpstimes
