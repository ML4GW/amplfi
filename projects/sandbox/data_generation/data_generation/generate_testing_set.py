import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import h5py
import numpy as np
import torch
from data_generation.utils import download_data, inject_into_background
from mldatafind.segments import query_segments
from mlpe.injection import generate_gw
from mlpe.logging import configure_logging
from typeo import scriptify

from ml4gw.gw import compute_observed_strain, get_ifo_geometry


@scriptify
def main(
    ifos: List[str],
    state_flag: str,
    frame_type: str,
    channel: str,
    start: float,
    stop: float,
    sample_rate: float,
    prior: Callable,
    waveform: Callable,
    n_samples: int,
    kernel_length: float,
    waveform_duration: float,
    datadir: Path,
    logdir: Path,
    min_duration: float = 0,
    waveform_arguments: Optional[Dict] = None,
    parameter_conversion: Optional[Callable] = None,
    force_generation: bool = False,
    verbose: bool = False,
):
    """
    Generate test dataset of kernels by injecting waveforms
    into random, non-coincident background kernels

    Queries first contiguous segment of background between start and stop
    that satisifies `min_duration` and `state_flag` requirements.

    Then, a dataset of waveforms is generated from the given prior,
    and injected into kernels sampled non-coincidentally from the background.

    Args:
        ifos: List of interferometers
        state_flag: Flag used to find times with good data quality.
        frame_type: Frame type used for discovering data
        channel: Channel for reading data
        start: Start time for finding data
        stop: Stop time for finding data
        sample_rate: Rate at which timeseries are sampled
        prior: Callable that instantiates a bilby prior
        waveform: A callable compatible with bilby waveform generator
        n_samples: Number of waveforms to sample and inject
        kernel_length: Length in seconds of kernels produced
        waveform_duration: length of injected waveforms
        datadir: Path to store data
        logdir: Path to store logs
        min_duration: Minimum duration of segments
        waveform_arguments:
            Additional arguments to pass to waveform generator,
            that will ultimately get passed
            to the waveform callable specified. For example,
            generating BBH waveforms requires the specification of a
            waveform_approximant
        parameter_conversion:
            Parameter conversion to pass the bilby waveform generator.
            Typically used for converting between bilby and lalsimulation
            BBH parameters
        force_generation: Force generation of data
        verbose: Log verbosely

    Returns signal file containiing injections and parameters
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

    # query the first coincident contiguous segment
    # of required minimum duration and download the data
    segment_names = [f"{ifo}:{state_flag}" for ifo in ifos]
    segment_start, segment_stop = query_segments(
        segment_names,
        start,
        stop,
        min_duration,
    )[0]

    background_dict = download_data(
        ifos, frame_type, channel, sample_rate, segment_start, segment_stop
    )

    background = np.stack([ts.value for ts in background_dict.values()])
    background = torch.as_tensor(background)

    # instantiate prior, sample, and generate signals
    prior = prior()
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
    plus = torch.Tensor(plus)
    cross = torch.Tensor(cross)

    kernel_size = int(kernel_length * sample_rate)

    # project raw polarizations onto interferometers
    # with sampled sky localizations
    tensors, vertices = get_ifo_geometry(*ifos)

    dec = torch.Tensor(parameters["dec"])
    psi = torch.Tensor(parameters["dec"])
    ra = torch.Tensor(parameters["dec"])

    waveforms = compute_observed_strain(
        dec,
        psi,
        ra,
        tensors,
        vertices,
        sample_rate,
        plus=plus,
        cross=cross,
    )

    # inject signals into randomly sampled kernels of background
    injections = inject_into_background(
        background,
        waveforms,
        kernel_size,
    )

    # write signals and parameters used to generate them
    with h5py.File(signal_file, "w") as f:

        f.create_dataset("injections", data=injections)

        for name, value in parameters.items():
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