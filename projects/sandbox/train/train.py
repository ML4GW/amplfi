import logging
from math import pi
from pathlib import Path
from typing import Callable, List, Optional

import h5py
import numpy as np
import torch
from utils import split
from validation import make_validation_dataset

from ml4gw.distributions import Cosine, LogUniform, Uniform
from ml4gw.transforms import ChannelWiseScaler
from ml4gw.waveforms import SineGaussian
from mlpe.architectures import embeddings, flows
from mlpe.data.dataloader import PEInMemoryDataset
from mlpe.data.transforms import Preprocessor
from mlpe.data.transforms.injection import PEInjector
from mlpe.logging import configure_logging
from mlpe.trainer import optimizers, schedulers, train
from typeo import scriptify
from typeo.utils import make_dummy


class ParameterSampler(torch.nn.Module):
    def __init__(self, **parameters: Callable):
        super().__init__()
        self.parameters = parameters

    def forward(
        self,
        N: int,
        device: str = "cpu",
    ):

        parameters = {k: v(N).to(device) for k, v in self.parameters.items()}
        return parameters


def load_background(background_path: Path, ifos):
    background = []
    with h5py.File(background_path) as f:
        for ifo in ifos:
            hoft = f[ifo][:]
            background.append(hoft)
    return np.stack(background)


def load_signals(waveform_dataset: Path, parameter_names: List[str]):
    """
    Load in signals and intrinsic parameters.
    If no intrinsic parameters are requested, return None.
    """
    with h5py.File(waveform_dataset, "r") as f:
        signals = f["signals"][:]

        # TODO: how do we ensure order
        # of parameters throughout pipeline?
        intrinsic = []
        for param in parameter_names:
            if param not in ["dec", "psi", "phi"]:
                values = f[param][:]
                # take logarithm since hrss
                # spans large magnitude range
                if param == "hrss":
                    values = np.log(values)
                intrinsic.append(values)

        if intrinsic:
            intrinsic = np.row_stack(intrinsic)
        else:
            intrinsic = None

    return signals, intrinsic


@scriptify(
    kwargs=make_dummy(
        train,
        exclude=[
            "flow",
            "embedding",
            "optimizer",
            "scheduler",
            "train_dataset",
            "valid_dataset",
            "preprocessor",
        ],
    ),
    flow=flows,
    embedding=embeddings,
    optimizer=optimizers,
    scheduler=schedulers,
)
def main(
    background_dataset: Path,
    waveform_dataset: Path,
    waveform_duration: float,
    flow: Callable,
    embedding: Callable,
    optimizer: Callable,
    scheduler: Callable,
    inference_params: List[str],
    ifos: List[str],
    sample_rate: float,
    kernel_length: float,
    fduration: float,
    batches_per_epoch: int,
    batch_size: int,
    device: str,
    outdir: Path,
    logdir: Path,
    valid_frac: Optional[float] = None,
    valid_stride: Optional[float] = None,
    verbose: bool = False,
    **kwargs
):

    logdir.mkdir(exist_ok=True, parents=True)
    configure_logging(logdir / "train.log", verbose)

    param_dim = len(inference_params)
    n_ifos = len(ifos)

    # load in background of shape (n_ifos, n_samples) and split into training
    # and validation if valid_frac specified
    background = load_background(background_dataset, ifos)

    logging.info(
        "Loading signals, performing train/val split, and preparing augmentors"
    )

    # load in the fixed set of validation waveforms
    # and split background into trainind and validation segments
    valid_signals, valid_intrinsic = load_signals(
        waveform_dataset, inference_params
    )
    if valid_frac is not None:
        background, valid_background = split(background, 1 - valid_frac, 1)

    # TODO: parameterize this somehow
    dec = Cosine()
    psi = Uniform(0, pi)
    phi = Uniform(-pi, pi)

    # intrinsic parameter sampler
    parameter_sampler = ParameterSampler(
        frequency=Uniform(32, 1024),
        quality=Uniform(2, 100),
        hrss=LogUniform(1e-23, 1e-19),
        phase=Uniform(0, 2 * pi),
        eccentricity=Uniform(0, 1),
    )

    # prepare waveform injector
    waveform = SineGaussian(
        sample_rate=sample_rate, duration=waveform_duration, device=device
    )

    # prepare injector
    injector = PEInjector(
        sample_rate,
        ifos,
        parameter_sampler,
        dec,
        psi,
        phi,
        waveform,
    )

    parameter_sampler.to(device)
    waveform.to(device)
    injector.to(device)

    # sample parameters from parameter sampler
    # so we can fit the standard scaler
    samples = parameter_sampler(100000)
    samples["dec"] = dec(100000)
    samples["psi"] = psi(100000)
    samples["phi"] = phi(100000)

    parameters = []
    for param in inference_params:
        values = samples[param]
        if param == "hrss":
            values = np.log(values)
        parameters.append(values)
    parameters = np.row_stack(parameters)

    standard_scaler = ChannelWiseScaler(param_dim)
    preprocessor = Preprocessor(
        n_ifos,
        sample_rate,
        fduration,
        scaler=standard_scaler,
    )

    preprocessor.scaler.fit(parameters)
    preprocessor.scaler.to(device)

    # create preprocessor out of whitening transform
    # for strain data, and standard scaler for parameters
    preprocessor.whitener.fit(kernel_length, *background)
    preprocessor.whitener.to(device)

    # TODO: this light preprocessor wrapper can probably be removed
    # save preprocessor
    preprocess_dir = outdir / "preprocessor"
    preprocess_dir.mkdir(exist_ok=True, parents=True)
    torch.save(
        preprocessor.whitener.state_dict(), preprocess_dir / "whitener.pt"
    )
    torch.save(preprocessor.scaler.state_dict(), preprocess_dir / "scaler.pt")

    # create full training dataloader that will sample
    # kernels from background, generate sine gaussian waveforms
    # and inject them into the background
    train_dataset = PEInMemoryDataset(
        background,
        int(kernel_length * sample_rate),
        batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
        preprocessor=injector,
        coincident=False,
        shuffle=True,
        device=device,
    )

    logging.info("Constructing validation dataloader")
    # construct validation dataset
    valid_dataset = None
    if valid_frac is not None:
        valid_dataset = make_validation_dataset(
            valid_background,
            injector,
            10000,
            ifos,
            dec,
            psi,
            phi,
            kernel_length,
            valid_stride,
            sample_rate,
            batch_size,
            device,
        )

    logging.info("Launching training")
    train(
        flow,
        embedding,
        optimizer,
        scheduler,
        outdir,
        train_dataset,
        valid_dataset,
        preprocessor,
        device=device,
        **kwargs
    )
