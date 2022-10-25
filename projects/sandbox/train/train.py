from math import pi
from pathlib import Path
from typing import List

import h5py
import numpy as np
from mlpe.data.dataloader import PEInMemoryDataset
from mlpe.data.distributions import Cosine, Uniform
from mlpe.data.transforms import (
    Preprocessor,
    StandardScalerTransform,
    WaveformInjector,
)
from mlpe.logging import configure_logging
from mlpe.trainer import trainify

EXTRINSIC_PARAMS = ["dec", "psi", "phi", "snr"]
# TODO: how to generalize this to be able
# to pass arbitrary distributions as function argument
EXTRINSIC_DISTS = {
    # uniform on sky
    "dec": Cosine(),
    "psi": Uniform(0, pi),
    "phi": Uniform(-pi, pi),
}


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
        plus, cross = signals.transpose(1, 0, 2)

        # TODO: how do we ensure order
        # of parameters throughout pipeline?
        data = []
        for param in parameter_names:
            if param not in EXTRINSIC_PARAMS:
                values = f[param][:]
                # take logarithm since hrss
                # spans large magnitude range
                if param == "hrss":
                    values = np.log10(values)
                data.append(values)

        intrinsic = np.column_stack(data)

    return plus, cross, intrinsic


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
    verbose: bool = False,
    **kwargs
):

    configure_logging(logdir / "train.log", verbose)
    num_ifos = len(ifos)
    num_params = len(inference_params)

    # load in background
    background = load_background(background_path, ifos)

    plus, cross, intrinsic = load_signals(waveform_dataset, inference_params)

    # prepare injector
    injector = WaveformInjector(
        sample_rate,
        ifos,
        dec=EXTRINSIC_DISTS["dec"],
        psi=EXTRINSIC_DISTS["psi"],
        phi=EXTRINSIC_DISTS["phi"],
        intrinsic_parameters=intrinsic,
        trigger_offset=trigger_distance,
        plus=plus,
        cross=cross,
    )

    injector.to(device)

    # construct samples of extrinsic parameters
    # if they were passed as inference params
    # so they can be fit to standard scaler
    n_signals = len(plus)

    for param, dist in EXTRINSIC_DISTS.items():
        if param in inference_params:
            samples = dist(n_signals)
            intrinsic = np.column_stack([intrinsic, samples])

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

    # create preprocessor
    # out of whitening transform
    # for strain data,
    # and standard scaler
    # for parameters

    standard_scaler = StandardScalerTransform(num_params)
    preprocessor = Preprocessor(
        num_ifos,
        sample_rate,
        kernel_length,
        normalizer=standard_scaler,
        fduration=fduration,
        highpass=highpass,
    )

    preprocessor.whitener.fit(background)
    preprocessor.whitener.to(device)

    preprocessor.normalizer.fit(intrinsic)
    preprocessor.normalizer.to(device)

    # TODO: Validation
    valid_dataset = None
    return train_dataset, valid_dataset, preprocessor
