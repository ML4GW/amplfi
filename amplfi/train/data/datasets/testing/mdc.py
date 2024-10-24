from pathlib import Path

import h5py
import numpy as np
import torch
from astropy.time import Time

from amplfi.train.data.datasets import FlowDataset


def phi_from_ra(ra: np.ndarray, gpstimes: np.ndarray) -> float:

    # get the sidereal time at the observation time
    gmsts = []
    for t in gpstimes:
        t = Time(t, format="gps", scale="utc")
        gmst = t.sidereal_time("mean", "greenwich").to("rad").value
        gmsts.append(gmst)

    # calculate the relative azimuthal angle in the range [0, 2pi]
    phi = ra - gmst
    mask = phi < 0
    phi[mask] += 2 * np.pi

    # convert phi from range [0, 2pi] to [-pi, pi]
    mask = phi > np.pi
    phi[mask] -= 2 * np.pi

    return phi


class MDCDataset(FlowDataset):
    """
    Testing dataset for iterating over MDC data
    """

    def __init__(self, mdc_file: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mdc_file = mdc_file
        self.i = 0

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        # load in the strain data and parameters
        strain = []
        parameters = {}
        with h5py.File(self.mdc_file, "r") as f:
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
        post = (
            self.waveform_sampler.padding
            + self.waveform_sampler.ringdown_duration
        )
        pre = (
            post
            - self.hparams.kernel_length
            - (self.hparams.fduration // 2)
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
            raise ValueError("Use of the MDC dataset is for testing only")

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
