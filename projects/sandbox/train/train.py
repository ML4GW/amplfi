from math import pi
from pathlib import Path
from typing import List

import h5py
import numpy as np
from mlpe.data.dataloader import PEInMemoryDataset
from mlpe.data.distributions import Cosine, Uniform
from mlpe.data.transforms import Preprocessor, StandardScalerTransform
from mlpe.trainer import trainify

from ml4gw.transforms import RandomWaveformInjection


def load_background(background: Path, *ifos):
    background = []
    with h5py.File(background) as f:
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
            data.append(f[param][:])

        parameters = np.column_stack(data)

    return plus, cross, parameters


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
):

    num_ifos = len(ifos)
    num_params = len(inference_params)

    # load in background
    background = load_background(background_path, ifos)

    plus, cross, parameters = load_signals(waveform_dataset, inference_params)

    # prepare injector
    injector = RandomWaveformInjection(
        dec=Cosine(),
        psi=Uniform(0, pi),
        phi=Uniform(-pi, pi),
        sample_rate=sample_rate,
        intrinsic_parameters=parameters,
        trigger_offset=trigger_distance,
        plus=plus,
        cross=cross,
    )

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

    # fit standard scaler to parameters
    standard_scaler = StandardScalerTransform(num_params)
    standard_scaler.fit(parameters)

    # create preprocessor
    # out of whitening transform
    # for strain data,
    # and standard scaler
    # for parameters
    preprocessor = Preprocessor(
        num_ifos,
        sample_rate,
        kernel_length,
        normalizer=standard_scaler,
        fduration=fduration,
        highpass=highpass,
    )

    # TODO: Validation
    return train_dataset, preprocessor
