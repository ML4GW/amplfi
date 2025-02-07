"""
Additional Lightning DataModules for testing
amplfi models across various usecases
"""
from pathlib import Path

import h5py
import numpy as np
import torch
from astropy.time import Time
from ml4gw.dataloading import Hdf5TimeSeriesDataset

from amplfi.train.data.datasets import FlowDataset

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
            data should be stored in an hdf5 dataset named
            after that interferometer.The dataset should be
            of shape (batch, time). It is assumed that the coalescence
            time of the injection is placed in the middle of each sample
            of the array. In addition, the parameters used to generate
            each injection should live in a dataset named after the
            parameter, e.g. `chirp_mass`.


    """

    def __init__(self, dataset_path: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_path = dataset_path
        self.i = 0

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        # load in the strain data and parameters
        strain = []
        parameters = {}
        with h5py.File(self.dataset_path, "r") as f:
            for ifo in self.hparams.ifos:
                strain.append(torch.tensor(f[ifo][:], dtype=torch.float32))

            for parameter in self.hparams.inference_params + ["ra", "gpstime"]:
                # skip phi since these are proper injections
                # where we'll need to convert ra to phi
                # given the gpstime
                if parameter == "phi":
                    continue

                parameters[parameter] = torch.tensor(
                    f[parameter][:], dtype=torch.float32
                )

        # convert ra to phi
        parameters["phi"] = torch.tensor(
            phi_from_ra(
                parameters["ra"].numpy(), parameters["gpstime"].numpy()
            )
        )
        strain = torch.stack(strain, dim=1)

        # based on psd length, fduration and kernel length, and padding,
        # determine slice indices. It is assumed the coalescence
        # time of the waveform is in the middle
        middle = strain.shape[-1] // 2
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

        # convert parameters to a tensor
        parameters = [
            torch.Tensor(parameters[k]) for k in self.hparams.inference_params
        ]
        parameters = torch.vstack(parameters).T

        # build dataset and dataloader that will
        # simply load one injection (and its parameters) at a time
        dataset = torch.utils.data.TensorDataset(strain, parameters)

        return torch.utils.data.DataLoader(
            dataset, batch_size=1, num_workers=12
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

        return X, asds, parameters


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
            to inject.
    """

    def __init__(self, dataset_path: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_path = dataset_path
        self.i = 0

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        parameters = {}

        with h5py.File(self.dataset_path) as f:
            for parameter in self.hparams.inference_params + ["ra", "gpstime"]:
                # skip phi since these are proper injections
                # where we'll need to convert ra to phi
                # given the gpstime
                if parameter == "phi":
                    continue

                parameters[parameter] = torch.tensor(
                    f[parameter][:], dtype=torch.float32
                )

        # apply conversion function to parameters
        waveform_parameters = (
            self.waveform_sampler.parameter_sampler.conversion_function(
                **parameters
            )
        )

        # convert ra to phi
        parameters["phi"] = torch.tensor(
            phi_from_ra(
                parameters["ra"].numpy(), parameters["gpstime"].numpy()
            )
        )

        # generate cross and plus using our infrastructure
        cross, plus = self.waveform_sampler(**waveform_parameters)

        params = []
        for k in self.hparams.inference_params:
            if k in parameters.keys():
                params.append(torch.Tensor(parameters[k]))

        parameters = torch.column_stack(params)

        waveform_dataset = torch.utils.data.TensorDataset(
            cross, plus, parameters
        )

        waveform_dataloader = torch.utils.data.DataLoader(
            waveform_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=10,
        )

        # TODO: add ability to inject at
        # specific times into the strain data
        background_dataset = Hdf5TimeSeriesDataset(
            self.test_fnames,
            channels=self.hparams.ifos,
            kernel_size=int(self.hparams.sample_rate * self.sample_length),
            batch_size=1,
            batches_per_epoch=len(waveform_dataloader),
            coincident=True,
        )

        background_dataloader = torch.utils.data.DataLoader(
            background_dataset, pin_memory=False, num_workers=10
        )
        return ZippedDataset(
            waveform_dataloader,
            background_dataloader,
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
                "Use of ParameterTestingDataset is for testing only"
            )

        [cross, plus, parameters], [background] = batch
        keys = [
            k
            for k in self.hparams.inference_params
            if k not in ["dec", "psi", "phi"]
        ]
        parameters = {k: parameters[:, i] for i, k in enumerate(keys)}
        strain, asds, parameters = self.inject(
            background, cross, plus, parameters
        )

        return strain, asds, parameters
