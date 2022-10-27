import logging
from math import pi
from pathlib import Path
from typing import Callable, Dict, List, Optional

import bilby
import h5py
import numpy as np
import torch
from mlpe.data.distributions import Cosine, Uniform
from mlpe.injection import generate_gw
from mlpe.logging import configure_logging
from typeo import scriptify

from ml4gw.gw import compute_observed_strain, get_ifo_geometry
from ml4gw.utils.slicing import slice_kernels

# TODO: this is used in train project
# should this be moved to injection library?
EXTRINSIC_DISTS = {
    # uniform on sky
    "dec": Cosine(),
    "psi": Uniform(0, pi),
    "phi": Uniform(-pi, pi),
}


def inject_into_background(
    background: np.ndarray,
    ifos: List[str],
    prior: bilby.gw.prior.PriorDict,
    waveform: Callable,
    sample_rate: float,
    n_samples: int,
    kernel_size: int,
    waveform_duration: float,
    waveform_arguments: Optional[Dict] = None,
    parameter_conversion: Optional[Callable] = None,
):

    parameters = prior.sample(n_samples)

    signals = generate_gw(
        parameters,
        sample_rate,
        waveform_duration,
        waveform,
        waveform_arguments=waveform_arguments,
        parameter_conversion=parameter_conversion,
    )

    plus, cross = signals.transpose(1, 0, 2)

    # project raw signals onto interferometers
    tensors, vertices = get_ifo_geometry(*ifos)

    waveforms = compute_observed_strain(
        torch.Tensor(parameters["dec"]),
        torch.Tensor(parameters["psi"]),
        torch.Tensor(parameters["ra"]),
        tensors,
        vertices,
        sample_rate,
        plus=plus,
        cross=cross,
    )

    # select the kernel size around the center
    # of the waveforms
    center = waveforms.shape[-1] // 2
    start = center - (kernel_size // 2)
    stop = center + (kernel_size // 2)
    waveforms = waveforms[:, start:stop]

    # sample random, non-coincident background kernels
    num_kernels = len(plus)

    idx = torch.randint(
        num_kernels,
        size=(num_kernels, len(background)),
    )

    X = slice_kernels(background, idx, kernel_size)

    # inject waveforms into background
    X += waveforms

    return X, parameters


@scriptify
def main(
    prior: Callable,
    waveform: Callable,
    inference_parameters: List[str],
    sample_rate: float,
    n_samples: int,
    waveform_duration: float,
    datadir: Path,
    logdir: Path,
    waveform_arguments: Optional[Dict] = None,
    parameter_conversion: Optional[Callable] = None,
    force_generation: bool = False,
    verbose: bool = False,
):
    """Generates dataset of waveforms to test PE

    Args:

        prior_file: Path to prior for generating waveforms
        waveform: A callable compatible with bilby waveform generator
        sample_rate: sample rate for generating waveform
        n_samples: number of signal to inject
        waveform_duration: length of injected waveforms
        datadir: Path to store data
        logdir: Path to store logs
        waveform_arguments:
            Additional arguments to pass to waveform generator,
            that will ultimately get passed
            to the waveform callable specified. For example,
            generating BBH waveforms requires the specification of a
            waveform_approximant
    """

    configure_logging(logdir / "generate_testing_set.log", verbose)

    datadir.mkdir(exist_ok=True, parents=True)
    logdir.mkdir(exist_ok=True, parents=True)

    signal_file = datadir / "test_injections.h5"

    if signal_file.exists() and not force_generation:
        logging.info(
            "Signal data already exists and forced generation is off. "
            "Not generating testing signals."
        )
        return signal_file

    injections, parameters = inject_into_background(
        prior,
        waveform,
        sample_rate,
        n_samples,
        waveform_duration,
        waveform_arguments,
        parameter_conversion,
    )

    # write signals and parameters used to generate them
    with h5py.File(signal_file, "w") as f:

        f.create_dataset("injections", data=injections)

        for name, value in zip(inference_parameters, parameters):
            f.create_dataset(name, data=value)

        # write attributes
        f.attrs.update(
            {
                "size": n_samples,
                "sample_rate": sample_rate,
                "waveform_duration": waveform_duration,
                "waveform": waveform.__name__,
            }
        )
        if waveform_arguments is not None:
            f.attrs.update(waveform_arguments)

    return signal_file


if __name__ == "__main__":
    main()
