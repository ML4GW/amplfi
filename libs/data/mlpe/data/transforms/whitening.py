from typing import Optional

import numpy as np
import torch
from gwpy.signal.filter_design import fir_from_transfer
from gwpy.timeseries import TimeSeries
from mlpe.data.transforms.transform import Transform

# fftlength is length of segments
# for calculating asd
DEFAULT_FFTLENGTH = 2


class WhiteningTransform(Transform):
    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        fftlength: float = DEFAULT_FFTLENGTH,
        highpass: Optional[float] = None,
        fduration: Optional[float] = None,
    ) -> None:
        """Torch module for performing whitening. The first and last
        (fduration / 2) seconds of data are corrupted by the whitening
        and will be cropped. Thus, the output length
        that is ultimately passed to the network will be
        (kernel_length - fduration)
        """

        super().__init__()
        self.num_ifos = num_ifos
        self.sample_rate = sample_rate
        self.kernel_length = kernel_length
        self.fftlength = fftlength

        self.df = 1 / kernel_length
        self.ncorner = int(highpass / self.df) if highpass else 0

        # TODO: is this the best default behavior
        self.fduration = fduration or kernel_length / 2

        # number of samples of corrupted data
        # due to settling in of whitening filter
        # TODO: should this be a parameter?
        self.crop_samples = int((self.fduration / 2) * self.sample_rate)
        self.ntaps = int(self.fduration * self.sample_rate)
        self.kernel_size = int(kernel_length * sample_rate)

        # subtract one to make kernel_size odd since the last value
        # of the filter will be 0. anyway. TODO: should we check
        # to confirm kernel_size is even first? Does that 0. still
        # get appended in the odd case?

        # initialize the parameter with 0s, then fill it out later
        self.time_domain_filter = self.add_parameter(
            torch.zeros((num_ifos, 1, self.ntaps - 1)),
        )

        # TODO: does this need to be a parameter?
        self.pad = int(self.time_domain_filter.size(-1) // 2)
        self.window = torch.hann_window(self.ntaps)

    def to(self, device: torch.device):
        """
        Quick override of device placement to ensure
        that our window, which is a _tensor_ and not
        a _parameter_, gets moved to the proper device
        """

        # explicitly send all objects used
        # in forward pass to specified device

        # TODO: pass device as argument to preprocessor
        # and add these all as parameters, passing the
        # device along?
        super().to(device)
        self.window = self.window.to(device)
        return self

    def fit(self, X: np.ndarray) -> None:
        """
        Build a whitening time domain filter from a set
        of ASDs. TODO: should this be a single tensor
        stacking all the backgrounds for consistency?
        """
        if X.ndim != 2:
            raise ValueError(
                "Expected background used to fit WhiteningTransform "
                "to have 2 dimensions, but found {}".format(X.ndim)
            )
        if len(X) != self.time_domain_filter.shape[0]:
            raise ValueError(
                "Expected to fit whitening transform on {} backgrounds, "
                "but was passed {}".format(
                    self.time_domain_filter.shape[0], len(X)
                )
            )

        tdfs = []
        for x in X:
            ts = TimeSeries(x, dt=1 / self.sample_rate)
            asd = ts.asd(
                fftlength=self.fftlength, window="hanning", method="median"
            )
            asd = asd.interpolate(self.df).value
            if (asd == 0).any():
                raise ValueError("Found 0 values in background asd")
            tdf = fir_from_transfer(
                1 / asd,
                ntaps=self.ntaps,
                window="hanning",
                ncorner=self.ncorner,
            )
            tdfs.append(tdf)

        tdf = np.stack(tdfs)[:, None, :-1]
        self.set_value(self.time_domain_filter, tdf)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # do a constant detrend along the time axis,
        X = X - X.mean(axis=-1, keepdims=True)
        X[:, :, : self.pad] *= self.window[: self.pad]
        X[:, :, -self.pad :] *= self.window[-self.pad :]

        # convolve the detrended data with the time-domain
        # filters constructed during initialization from
        # the background data, using groups to ensure that
        # the convolution is performed independently for
        # each interferometer channel

        # see
        # https://github.com/gwpy/gwpy/blob/main/gwpy/timeseries/timeseries.py
        # for logic

        nfft = min(8 * self.time_domain_filter.size(-1), self.kernel_size)

        if nfft >= self.kernel_size / 2:
            conv = torch.nn.functional.conv1d(
                X,
                self.time_domain_filter,
                groups=self.num_ifos,
                padding=int(self.pad),
            )

            # crop the beginning and ending fduration / 2
            conv = conv[:, :, self.crop_samples : -self.crop_samples]

        # TODO: speed this up using torch.unfold
        # and removing while loop

        # else use the overlap-save algorithm
        else:
            raise NotImplementedError(
                "An optimal torch implementation of whitening for short "
                "fdurations is not complete. Use a larger fduration "
            )
            nstep = nfft - 2 * self.pad
            conv = torch.zeros_like(X)
            # handle first chunk separately
            conv[:, :, : nfft - self.pad] = torch.nn.functional.conv1d(
                X[:, :, :nfft],
                self.time_domain_filter,
                groups=self.num_ifos,
                padding=self.pad,
            )[:, :, : nfft - self.pad]

            # process chunks of length nstep
            k = nfft - self.pad
            while k < self.kernel_size - nfft + self.pad:
                yk = torch.nn.functional.conv1d(
                    X[:, :, k - self.pad : k + nstep + self.pad],
                    self.time_domain_filter,
                    groups=self.num_ifos,
                    padding=self.pad,
                )
                conv[:, :, k : k + yk.size(-1) - 2 * self.pad] = yk[
                    :, :, self.pad : -self.pad
                ]
                k += nstep

            # handle last chunk separately
            conv[:, :, -nfft + self.pad :] = torch.nn.functional.conv1d(
                X[:, :, -nfft:],
                self.time_domain_filter,
                groups=self.num_ifos,
                padding=self.pad,
            )[:, :, -nfft + self.pad :]

            # crop the beginning and ending fduration / 2
            conv = conv[:, :, self.crop_samples : -self.crop_samples]

        # scale by sqrt(2 / sample_rate) for some inscrutable
        # signal processing reason beyond my understanding

        return conv * (2 / self.sample_rate) ** 0.5
