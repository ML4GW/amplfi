import logging
import os
import sys
from typing import Dict, List, Optional, Sequence

import h5py
import lightning.pytorch as pl
import torch
from ml4gw.dataloading import Hdf5TimeSeriesDataset, InMemoryDataset
from ml4gw.transforms import ChannelWiseScaler, Whiten

from ...augmentations import PsdEstimator, WaveformProjector
from ..utils import fs as fs_utils
from ..utils.utils import ZippedDataset
from amplfi.train.prior import ParameterTransformer
from ..waveforms.sampler import WaveformSampler
import numpy as np
from pathlib import Path
import random
from tqdm.auto import tqdm

Tensor = torch.Tensor
Distribution = torch.distributions.Distribution

SECONDS_PER_DAY = 86400


class AmplfiDataset(pl.LightningDataModule):
    """
    Base LightningDataModule for loading data to train AMPLFI models.

    Subclasses must define the `inject` method
    which encodes how background strain,
    cross/plus polarizations and parameters
    are processed before being passed to a model

    Args:
        data_dir:
            Path to directory containing training and testing data
        inference_params:
            List of parameters to perform inference on. Can be a subset
            of the parameters that fully describes the waveforms
        dec:
            The distribution of declinations to sample from
        psi:
            The distribution of polarization angles to sample from
        phi:
            The distribution of "right ascensions" to sample from
        highpass:
            Highpass frequency in Hz
        sample_rate:
            Rate data is sampled in Hz
        kernel_length:
            Length of the kernel seen by model in seconds
        fduration:
            The length of the whitening filter's impulse
            response, in seconds. `fduration / 2` seconds
            worth of data will be cropped from the edges
            of the whitened timeseries.
        psd_length:
            Length of data used to calculate psd in seconds
        batches_per_epoch:
            Number of batches for each training epoch.
        batch_size:
            Number of samples in each batch
        ifos:
            List of interferometers
        waveform_sampler:
            `WaveformSampler` object that produces waveforms and parameters
            for training, validation and testing.
            See `train.data.waveforms.sampler`
            for methods this object should define.
        parameter_transformer:
            A `ParameterTransformer` object that applies any
            additional transformations to parameters before
            they are scaled and passed to the neural network.
        train_val_range:
            Tuple of gpstimes that specify time range of
            training and validation data.
            Will filter the data directory to only include files that contain
            segments that overlap with this range. If `None`, use all data.
        test_range:
            Tuple of gpstimes that specify range of testing data to use.
            Will filter the data directory to only include files that contain
            segments that overlap with this range. If `None`, use all data.
        fftlength:
            Length of the fft used to calculate the psd.
            Defaults to `kernel_length`
        min_valid_duration:
            Minimum number of seconds of validation background data
        num_files_per_batch:
            Number of strain hdf5 files to use to construct
            each batch. Can lead to dataloading performance increases.
        max_num_workers:
            Maximum number of workers to assign to each
            training dataloader.

    """

    def __init__(
        self,
        data_dir: str,
        inference_params: list[str],
        dec: Distribution,
        psi: Distribution,
        phi: Distribution,
        highpass: float,
        sample_rate: float,
        kernel_length: float,
        fduration: float,
        psd_length: float,
        batches_per_epoch: int,
        batch_size: int,
        ifos: List[str],
        waveform_sampler: WaveformSampler,
        parameter_transformer: Optional[ParameterTransformer] = None,
        fftlength: Optional[int] = None,
        train_val_range: Optional[tuple[float, float]] = None,
        test_range: Optional[tuple[float, float]] = None,
        min_valid_duration: float = 10000,
        num_files_per_batch: Optional[int] = None,
        max_num_workers: int = 6,
        verbose: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__()
        # TODO: we shouldn't have to do this, since
        # lightning calls this in the `trainer`, but
        # I found that our testing parameters were different
        # even when using the same seed
        pl.seed_everything(seed)
        self.save_hyperparameters(ignore=["waveform_sampler"])
        self.init_logging(verbose)
        self.waveform_sampler = waveform_sampler
        self.max_num_workers = max_num_workers

        self.dec, self.psi, self.phi = dec, psi, phi
        self.parameter_transformer = parameter_transformer or (lambda x: x)

        # generate our local node data directory
        # if our specified data source is remote
        self.data_dir = fs_utils.get_data_dir(self.hparams.data_dir)

    def init_logging(self, verbose: bool):
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            format=log_format,
            level=logging.DEBUG if verbose else logging.INFO,
            stream=sys.stdout,
        )

    def prepare_data(self):
        """
        Download s3 data if it doesn't exist.
        """
        logger = logging.getLogger(self.__class__.__name__)
        bucket, _ = fs_utils.split_data_dir(self.hparams.data_dir)
        if bucket is None:
            return
        logger.info(
            "Downloading data from S3 bucket {} to {}".format(
                bucket, self.data_dir
            )
        )
        fs_utils.download_training_data(bucket, self.data_dir)

    # ================================================ #
    # Distribution utilities
    # ================================================ #
    def get_world_size_and_rank(self) -> tuple[int, int]:
        """
        Name says it all, but generalizes to the case
        where we aren't running distributed training.
        """
        if not torch.distributed.is_initialized():
            return 1, 0
        else:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            return world_size, rank

    def get_logger(self, world_size, rank):
        logger_name = self.__class__.__name__
        if world_size > 1:
            logger_name += f":{rank}"
        return logging.getLogger(logger_name)

    @property
    def device(self):
        """Return the device of the associated lightning module"""
        return self.trainer.lightning_module.device

    # ================================================ #
    # Helper utilities for preprocessing
    # ================================================ #

    def transform(self, parameters: Dict[str, Tensor]):
        """
        Make transforms to parameters before scaling
        and performing training/inference.
        For example, taking logarithm of hrss
        """
        return self.parameter_transformer(parameters)

    def scale(self, parameters, reverse: bool = False):
        """
        Apply standard scaling to transformed parameters
        """
        parameters = parameters.transpose(1, 0)
        scaled = self.scaler(parameters, reverse=reverse)
        scaled = scaled.transpose(1, 0)
        return scaled

    # ================================================ #
    # Re-parameterizing some attributes
    # ================================================ #

    @property
    def frequencies(self):
        """
        Frequencies corresponding to an fft
        of a timeseries of length `kernel_length`
        """
        size = int(self.hparams.kernel_length * self.hparams.sample_rate)
        return torch.fft.rfftfreq(size, d=1 / self.hparams.sample_rate)

    @property
    def sample_length(self):
        return (
            self.hparams.kernel_length
            + self.hparams.fduration
            + self.hparams.psd_length
        )

    @property
    def num_ifos(self):
        return len(self.hparams.ifos)

    @property
    def num_params(self):
        return len(self.hparams.inference_params)

    @property
    def num_workers(self):
        local_world_size = len(self.trainer.device_ids)
        return min(
            self.max_num_workers, int(os.cpu_count() / local_world_size)
        )

    @property
    def val_batch_size(self):
        """Use larger batch sizes when we don't need gradients."""
        return int(1 * self.hparams.batch_size)

    def filter_fnames(self, fnames: Sequence[str], start: float, end: float):
        filtered_fnames = []
        fstarts = [int(fname.stem.split("-")[1]) for fname in fnames]

        for i, fstart in enumerate(fstarts):
            if fstart >= start and fstart <= end:
                filtered_fnames.append(fnames[i])
        return filtered_fnames

    def get_train_val_fnames(self):
        """List of background files used for both training and validation"""
        background_dir = self.data_dir / "train" / "background"
        fnames = sorted((background_dir.glob("*.hdf5")))
        if self.hparams.train_val_range is not None:
            start, end = self.hparams.train_val_range
            self._logger.info(
                f"Downselecting training and validation data "
                f"between {start} to {end}"
            )
            fnames = self.filter_fnames(fnames, start, end)

        return fnames

    def get_test_fnames(self):
        """List of background files used for testing a trained model"""
        test_dir = self.data_dir / "test" / "background"
        fnames = list(test_dir.glob("*.hdf5"))
        if self.hparams.test_range is not None:
            start, end = self.hparams.test_range
            self._logger.info(
                f"Downselecting testing data between {start} to {end}"
            )
            fnames = self.filter_fnames(fnames, start, end)

        duration = (
            sum([int(fname.stem.split("-")[-1]) for fname in fnames])
            / SECONDS_PER_DAY
        )
        self._logger.info(
            f"Using {len(fnames)} files with a total duration "
            f"of {duration:.3f} days for testing"
        )
        return fnames

    def train_val_split(self) -> Sequence[str]:
        """
        Split background files into training and validation sets
        based on the requested duration of the validation set
        """
        fnames = sorted(self.get_train_val_fnames())

        durations = [int(fname.stem.split("-")[-1]) for fname in fnames]
        valid_fnames = []
        valid_duration = 0
        while valid_duration < self.hparams.min_valid_duration:
            fname, duration = fnames.pop(-1), durations.pop(-1)
            valid_duration += duration
            valid_fnames.append(str(fname))

        train_fnames = fnames
        train_duration = (
            sum([int(fname.stem.split("-")[-1]) for fname in train_fnames])
            / SECONDS_PER_DAY
        )

        self._logger.info(
            f"Using {len(train_fnames)} files with a total duration "
            f"of {train_duration:.3f} days for training"
        )
        self._logger.info(
            f"Using {len(valid_fnames)} files with a total duration "
            f"of {valid_duration / SECONDS_PER_DAY:.3f} days for validation"
        )
        return train_fnames, valid_fnames

    # ================================================ #
    # Utilities for initial data loading and preparation
    # ================================================ #

    def transforms_to_device(self):
        """
        Move all `torch.nn.Modules` to the local device
        """
        for item in self.__dict__.values():
            if isinstance(item, torch.nn.Module):
                item.to(self.device)

    def build_transforms(self, stage):
        """
        Build torch.nn.Modules that will be used for on-device
        augmentation and preprocessing. Transfer these modules
        to the appropiate device
        """
        self._logger.info("Building torch Modules and transferring to device")
        window_length = self.hparams.kernel_length + self.hparams.fduration
        fftlength = self.hparams.fftlength or window_length
        self.psd_estimator = PsdEstimator(
            window_length,
            self.hparams.sample_rate,
            fftlength,
            fast=self.hparams.highpass is not None,
            average="median",
        )

        self.whitener = Whiten(
            self.hparams.fduration,
            self.hparams.sample_rate,
            self.hparams.highpass,
        )

        # build standard scaler object and fit to parameters;
        # waveform_sampler subclasses will decide how to generate
        # parameters to fit the scaler.
        # Only fit the scaler during training.
        # During testing, use pre-fit parameters
        if stage in ["predict", "test"] and self.trainer.model.scaler.built:
            self._logger.info("Using pre-fit standard scaler")
            self.scaler = self.trainer.model.scaler
        else:
            self._logger.info("Fitting standard scaler to parameters")
            self.scaler = self.fit_scaler()

        self.projector = WaveformProjector(
            self.hparams.ifos, self.hparams.sample_rate
        )

    def sample_extrinsic(self, X: torch.Tensor):
        """
        Sample extrinsic parameters used to project waveforms
        """
        N = len(X)
        dec = self.dec.sample((N,)).to(X.device)
        psi = self.psi.sample((N,)).to(X.device)
        phi = self.phi.sample((N,)).to(X.device)
        return dec, psi, phi

    def fit_scaler(self):
        scaler = ChannelWiseScaler(self.num_params)
        parameters = self.waveform_sampler.get_fit_parameters()
        key = list(parameters.keys())[0]
        dec, psi, phi = self.sample_extrinsic(parameters[key])
        parameters["dec"] = dec
        parameters["psi"] = psi
        parameters["phi"] = phi

        transformed = self.parameter_transformer(parameters)
        fit = []
        for key in self.hparams.inference_params:
            fit.append(transformed[key])

        fit = torch.row_stack(fit)
        scaler.fit(fit)
        return scaler

    def setup(self, stage: str) -> None:
        world_size, rank = self.get_world_size_and_rank()
        self._logger = self.get_logger(world_size, rank)

        self._logger.info(f"Setting up data for stage {stage}")
        if stage in ["fit", "validate"]:
            self.train_fnames, self.val_fnames = self.train_val_split()

        elif stage == "test":
            self.test_fnames = self.get_test_fnames()

        # infer sample rate directly from background data
        # and validate that it matches specified sample rate
        sample_file = (
            self.train_fnames[0]
            if stage in ["fit", "validate"]
            else self.test_fnames[0]
        )
        with h5py.File(sample_file, "r") as f:
            sample_rate = 1 / f[self.hparams.ifos[0]].attrs["dx"]
            assert sample_rate == self.hparams.sample_rate

        self._logger.info(f"Inferred sample rate of {sample_rate} Hz")

        # load validation/testing waveforms, parameters, and background
        # and build the fixed background dataset while data and augmentations
        # modules are all still on CPU.
        # get_val_waveforms should be implemented by waveform_sampler object
        if stage in ["fit", "validate"]:
            self._logger.info("Loading waveforms for validation")
            cross, plus, parameters = self.waveform_sampler.get_val_waveforms(
                rank, world_size
            )
            self.val_background = self.load_val_background(cross.shape[0])
            self._logger.info(f"Loaded {len(cross)} waveforms for validation")
            params = []
            for k in self.hparams.inference_params:
                if k in parameters.keys():
                    params.append(torch.Tensor(parameters[k]))

            self.val_waveforms = torch.stack([cross, plus], dim=0)
            self.val_parameters = torch.column_stack(params)

        elif stage == "test":
            (
                cross,
                plus,
                parameters,
            ) = self.waveform_sampler.get_test_waveforms()

            self._logger.info(f"Loaded {len(cross)} waveforms for testing")

            params = []
            for k in self.hparams.inference_params:
                if k in parameters.keys():
                    params.append(torch.Tensor(parameters[k]))

            self.test_inference_params = torch.column_stack(params)
            self.test_parameters: dict[str, torch.tensor] = parameters
            self.test_waveforms = torch.stack([cross, plus], dim=0)

        # once we've generated validation/testing waveforms on cpu,
        # build data augmentation modules
        # and transfer them to appropiate device
        self.build_transforms(stage)
        self.transforms_to_device()

    def load_background(self, fnames: Sequence[str]):
        background = []
        for fname in fnames:
            data = []
            with h5py.File(fname, "r") as f:
                for ifo in self.hparams.ifos:
                    back = f[ifo][:]
                    data.append(torch.tensor(back, dtype=torch.float32))
            data = torch.stack(data)
            background.append(data)
        return background

    def load_val_background(self, N: int) -> List[Tensor]:
        """
        Sample `N` background segments from the set of validation files.
        """
        dataset = Hdf5TimeSeriesDataset(
            self.val_fnames,
            channels=self.hparams.ifos,
            kernel_size=int(self.hparams.sample_rate * self.sample_length),
            batch_size=N,
            batches_per_epoch=1,
            coincident=False,
        )
        background = next(iter(dataset))
        return background

    def on_after_batch_transfer(self, batch, _):
        """
        This is a Lightning `hook` that gets called after
        data returned by a dataloader gets put on the local device,
        but before it gets passed to model for inference.

        Use this to do on-device augmentation/preprocessing
        """
        if self.trainer.training:
            [batch] = batch
            cross, plus, parameters = self.waveform_sampler.sample(batch)
            strain, asds, parameters, snrs = self.inject(
                batch, cross, plus, parameters
            )

        elif self.trainer.validating or self.trainer.sanity_checking:
            [cross, plus, parameters], [background] = batch
            background = background[: len(cross)]
            keys = [
                k
                for k in self.hparams.inference_params
                if k not in ["dec", "psi", "phi"]
            ]
            parameters = {k: parameters[:, i] for i, k in enumerate(keys)}
            strain, asds, parameters, snrs = self.inject(
                background, cross, plus, parameters
            )

        elif self.trainer.testing:
            [cross, plus, parameters], [background] = batch
            keys = [
                k
                for k in self.hparams.inference_params
                if k not in ["dec", "psi", "phi"]
            ]
            parameters = {k: parameters[:, i] for i, k in enumerate(keys)}
            strain, asds, parameters, snrs = self.inject(
                background, cross, plus, parameters
            )

        return strain, asds, parameters, snrs

    # ================================================ #
    # Dataloaders used by lightning
    # ================================================ #

    def train_dataloader(self):
        # if we only have one training file
        # load it into memory and use InMemoryDataset
        if len(self.train_fnames) == 1:
            train_background = self.load_background(self.train_fnames)[0]
            dataset = InMemoryDataset(
                train_background,
                kernel_size=int(self.hparams.sample_rate * self.sample_length),
                batch_size=self.hparams.batch_size,
                coincident=False,
                batches_per_epoch=self.hparams.batches_per_epoch,
                shuffle=True,
            )
        else:
            dataset = Hdf5TimeSeriesDataset(
                self.train_fnames,
                channels=self.hparams.ifos,
                kernel_size=int(self.hparams.sample_rate * self.sample_length),
                batch_size=self.hparams.batch_size,
                batches_per_epoch=self.hparams.batches_per_epoch,
                coincident=False,
                num_files_per_batch=self.hparams.num_files_per_batch,
            )

        self._logger.info(
            f"Using a {dataset.__class__.__name__} class for training"
        )
        pin_memory = isinstance(
            self.trainer.accelerator, pl.accelerators.CUDAAccelerator
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=self.num_workers, pin_memory=pin_memory
        )
        return dataloader

    def val_dataloader(self):
        # build waveform dataloader
        cross, plus = self.val_waveforms
        waveform_dataset = torch.utils.data.TensorDataset(
            cross, plus, self.val_parameters
        )
        waveform_dataloader = torch.utils.data.DataLoader(
            waveform_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            pin_memory=False,
        )

        background_dataset = torch.utils.data.TensorDataset(
            self.val_background
        )

        background_dataloader = torch.utils.data.DataLoader(
            background_dataset,
            batch_size=self.val_batch_size,
            pin_memory=False,
        )
        return ZippedDataset(
            waveform_dataloader,
            background_dataloader,
        )

    def test_dataloader(self):
        # build waveform dataloader
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
        if len(self.test_fnames) == 1:
            test_background = self.load_background(self.test_fnames)[0]
            background_dataset = InMemoryDataset(
                test_background,
                kernel_size=int(self.hparams.sample_rate * self.sample_length),
                batch_size=self.hparams.batch_size,
                coincident=False,
                batches_per_epoch=self.hparams.batches_per_epoch,
                shuffle=True,
            )
        else:
            background_dataset = Hdf5TimeSeriesDataset(
                self.test_fnames,
                channels=self.hparams.ifos,
                kernel_size=int(self.hparams.sample_rate * self.sample_length),
                batch_size=1,
                batches_per_epoch=len(waveform_dataloader),
                coincident=False,
            )

        background_dataloader = torch.utils.data.DataLoader(
            background_dataset, pin_memory=False, num_workers=10
        )
        return ZippedDataset(
            waveform_dataloader,
            background_dataloader,
        )

    def inject(self, *args, **kwargs):
        """
        Subclasses should implement this method
        for different training use cases,
        like training a similarity embedding
        or training a normalizing flow. This is called
        after the data is transferred to the local device
        """
        raise NotImplementedError

    def background_from_gpstimes(
        self,
        gpstimes: np.ndarray,
        fnames: List[Path],
        use_random_segment: bool = True,
    ) -> torch.Tensor:
        """
        Construct a Tensor of background segments corresponding
        to requested `gpstimes` where `gpstimes` specifies
        the desired location of the coalescence time
        """

        # load in background segments corresponding to gpstimes
        background = []
        analyzed_gpstimes = []
        segments = [
            tuple(map(float, f.name.split(".")[0].split("-")[1:]))
            for f in fnames
        ]

        def find_file(time: float) -> Optional[Path]:
            """
            Find file that contains `time`
            """
            for i, (start, length) in enumerate(segments):
                in_segment = time > start + self.sample_length
                in_segment &= time < (start + length - self.sample_length)
                if in_segment:
                    return fnames[i], start
            else:
                return None, None

        # calculate seconds of data to query
        # before and after coalescence time
        post = self.waveform_sampler.right_pad + self.hparams.fduration / 2
        pre = (
            post
            - self.hparams.kernel_length
            - (self.hparams.fduration / 2)
            - self.hparams.psd_length
        )

        # convert to number of indices
        num_post = int(post * self.hparams.sample_rate)
        num_pre = int(pre * self.hparams.sample_rate)

        background = []
        self._logger.info("Loading background segments for testing")
        for time in tqdm(gpstimes):
            time = time.item()
            strain = []

            # find file for this gpstime
            file, start = find_file(time)
            # if none exists, use random segment
            if file is None:
                if use_random_segment:
                    self._logger.info(
                        "No segment in testing directory containing "
                        f"{time}. Using random segment"
                    )
                    file = random.choice(fnames)
                    start, length = list(
                        map(float, file.name.split(".")[0].split("-")[1:])
                    )
                    time = start + random.randint(
                        -int(pre // 1),
                        int(length - post),
                    )
                else:
                    self._logger.info(
                        "No segment in testing directory containing "
                        f"{time}. Not analyzing"
                    )
                    continue

            analyzed_gpstimes.append(time)
            # convert from time to index in file
            middle_idx = int((time - start) * self.hparams.sample_rate)
            start_idx = middle_idx + num_pre
            end_idx = middle_idx + num_post
            with h5py.File(file) as f:
                for ifo in self.hparams.ifos:
                    strain.append(f[ifo][start_idx:end_idx])
                strain = np.stack(strain, axis=0)
                background.append(strain)
        background = np.stack(background, axis=0)
        return torch.tensor(background), analyzed_gpstimes
