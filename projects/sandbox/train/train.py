import logging
from pathlib import Path
from typing import Callable, List, Optional

import h5py
import numpy as np
import torch
from utils import EXTRINSIC_DISTS, prepare_augmentation, split
from validation import make_validation_dataset

from ml4gw.transforms import ChannelWiseScaler
from mlpe.architectures import embeddings, flows
from mlpe.data.dataloader import PEInMemoryDataset
from mlpe.data.transforms import Preprocessor
from mlpe.logging import configure_logging
from mlpe.trainer import train
from typeo import scriptify
from typeo.utils import make_dummy


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
            if param not in EXTRINSIC_DISTS.keys():
                values = f[param][:]
                # take logarithm since hrss
                # spans large magnitude range
                if param == "hrss":
                    values = np.log10(values)
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
            "train_dataset",
            "valid_dataset",
            "preprocessor",
            "flow",
            "embedding",
        ],
    ),
    flow=flows,
    embedding=embeddings,
)
def main(
    background_path: Path,
    waveform_dataset: Path,
    flow: Callable,
    embedding: Callable,
    inference_params: List[str],
    ifos: List[str],
    sample_rate: float,
    trigger_distance: float,
    kernel_length: float,
    fduration: float,
    highpass: float,
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
    num_ifos = len(ifos)
    num_params = len(inference_params)

    # load in background of shape (n_ifos, n_samples) and split into training
    # and validation if valid_frac specified
    background = load_background(background_path, ifos)

    logging.info(
        "Loading signals, performing train/val split, and preparing augmentors"
    )
    # intrinsic parameters is an array of shape (n_params, n_signals)
    signals, intrinsic = load_signals(waveform_dataset, inference_params)

    if valid_frac is not None:
        background, valid_background = split(background, 1 - valid_frac, 1)

    # note: we pass the transpose the intrinsic parameters here because
    # the ml4gw transforms expects an array of shape (n_signals, n_params)

    injector, valid_injector = prepare_augmentation(
        signals,
        ifos,
        valid_frac,
        sample_rate,
        trigger_distance,
        highpass,
        intrinsic,
    )

    injector.to(device, waveforms=True)

    # construct samples of extrinsic parameters
    # if they were passed as inference params
    # so they can be fit to standard scaler.
    n_signals = len(signals)

    # if no intrinsic parameters are requested,
    # set parameters to empty list,
    # otherwise, set to list of intrinsic parameters
    parameters = []
    if intrinsic is not None:
        parameters = list(intrinsic)

    # append extrinsic parameters to list to be fit
    for param, dist in EXTRINSIC_DISTS.items():
        if param in inference_params:
            samples = dist(n_signals)
            parameters.append(samples)

    parameters = np.row_stack(parameters)

    # create full training dataloader
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

    logging.info("Preparing preprocessors")
    # create preprocessor out of whitening transform
    # for strain data, and standard scaler for parameters
    standard_scaler = ChannelWiseScaler(num_params)
    preprocessor = Preprocessor(
        num_ifos,
        sample_rate,
        fduration,
        scaler=standard_scaler,
    )

    preprocessor.whitener.fit(kernel_length, *background)
    preprocessor.whitener.to(device)

    # to perform the normalization over each parameters,
    # the ml4gw ChannelWiseScaler expects an array of shape
    # (n_params, n_signals), so we pass the untransposed
    # intrinsic parameters here
    preprocessor.scaler.fit(parameters)
    preprocessor.scaler.to(device)

    # TODO: this light preprocessor wrapper can probably be removed
    # save preprocessor
    preprocess_dir = outdir / "preprocessor"
    preprocess_dir.mkdir(exist_ok=True, parents=True)
    torch.save(preprocessor.whitener, preprocess_dir / "whitener.pt")
    torch.save(preprocessor.scaler, preprocess_dir / "scaler.pt")

    logging.debug("Constructing validation dataloader")
    # construct validation dataset
    # from validation injector
    valid_dataset = None
    if valid_frac is not None:
        valid_dataset = make_validation_dataset(
            valid_background,
            valid_injector,
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
        outdir,
        train_dataset,
        valid_dataset,
        preprocessor,
        device=device,
        **kwargs
    )
