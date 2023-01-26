from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import torch
from mlpe.data.dataloader import PEInMemoryDataset
from mlpe.data.transforms import Preprocessor
from mlpe.logging import configure_logging
from mlpe.trainer import trainify
from utils import EXTRINSIC_DISTS, prepare_augmentation, split
from validation import make_validation_dataset

from ml4gw.transforms import ChannelWiseScaler


def load_background(background_path: Path, ifos):
    background = []
    with h5py.File(background_path) as f:
        for ifo in ifos:
            hoft = f[ifo][:]
            background.append(hoft)
    return np.stack(background)


def load_signals(waveform_dataset: Path, parameter_names: List[str]):

    with h5py.File(waveform_dataset, "r") as f:
        signals = f["signals"][:]

        # TODO: how do we ensure order
        # of parameters throughout pipeline?
        data = []
        for param in parameter_names:
            if param not in EXTRINSIC_DISTS.keys():
                values = f[param][:]
                # take logarithm since hrss
                # spans large magnitude range
                if param == "hrss":
                    values = np.log10(values)
                data.append(values)

        intrinsic = np.row_stack(data)

    return signals, intrinsic


@trainify
def main(
    background_path: Path,
    waveform_dataset: Path,
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

    configure_logging(logdir / "train.log", verbose)
    num_ifos = len(ifos)
    num_params = len(inference_params)

    # load in background and split into training
    # and validation if valid_frac specified
    background = load_background(background_path, ifos)

    signals, intrinsic = load_signals(waveform_dataset, inference_params)

    if valid_frac is not None:
        background, valid_background = split(background, 1 - valid_frac, 1)

    injector, valid_injector = prepare_augmentation(
        signals,
        intrinsic,
        ifos,
        valid_frac,
        sample_rate,
        trigger_distance,
        highpass,
    )

    injector.to(device, waveforms=True)

    # construct samples of extrinsic parameters
    # if they were passed as inference params
    # so they can be fit to standard scaler
    n_signals = len(signals)

    for param, dist in EXTRINSIC_DISTS.items():
        if param in inference_params:
            samples = dist(n_signals)
            intrinsic = np.row_stack([intrinsic, samples])

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

    # create preprocessor out of whitening transform
    # for strain data, and standard scaler for parameters
    standard_scaler = ChannelWiseScaler(num_params)
    preprocessor = Preprocessor(
        num_ifos,
        sample_rate,
        fduration,
        normalizer=standard_scaler,
    )

    preprocessor.whitener.fit(background)
    preprocessor.whitener.to(device)

    preprocessor.normalizer.fit(intrinsic)
    preprocessor.normalizer.to(device)

    torch.save(preprocessor, outdir / "preprocessor.pt")

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

    return train_dataset, valid_dataset, preprocessor
