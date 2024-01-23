import logging
from pathlib import Path
from typing import Callable, List

import h5py
import numpy as np
import torch
from data_generation.utils import (
    download_data,
    inject_into_random_background,
    noise_from_psd,
)
from gwpy.timeseries import TimeSeries
from mldatafind.segments import query_segments

from ml4gw.gw import compute_observed_strain, get_ifo_geometry
from ml4gw.spectral import normalize_psd
from mlpe.injection import generate_time_domain_sine_gaussian
from mlpe.logging import configure_logging
from typeo import scriptify


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
    n_samples: int,
    kernel_length: float,
    waveform_duration: float,
    datadir: Path,
    logdir: Path,
    min_duration: float = 0,
    gaussian: bool = False,
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
        ifos:
            List of interferometers
        state_flag:
            Flag used to find times with good data quality.
        frame_type:
            Frame type used for discovering data
        channel:
            Channel for reading data
        start:
            Start time for finding data
        stop:
            Stop time for finding data
        sample_rate:
            Rate at which timeseries are sampled
        prior:
            Callable that instantiates a bilby prior
        waveform:
            A callable compatible with bilby waveform generator
        n_samples:
            Number of waveforms to sample and inject
        kernel_length:
            Length in seconds of kernels produced
        waveform_duration:
            length of injected waveforms
        datadir:
            Path to store data
        logdir:
            Path to store logs
        min_duration:
            Minimum duration of segments
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
        gaussian:
            If True, generate gaussian noise from a psd calculated using
            the requested period of data. This will be used as the background
            to inject the waveforms into.
        force_generation:
            Force generation of data
        verbose:
            Log verbosely

    Returns signal file containiing injections and parameters
    """

    configure_logging(logdir / "generate_testing_set.log", verbose)

    datadir.mkdir(exist_ok=True, parents=True)
    logdir.mkdir(exist_ok=True, parents=True)

    signal_file = datadir / "flow_injections.h5"

    if signal_file.exists() and not force_generation:
        logging.info(
            "PP plot injection data already exists and "
            "forced generation is off. Not generating PP plot signals."
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

    df = 1 / waveform_duration
    if gaussian:
        logging.info(
            "Generating gaussian noise from psd for injection background"
        )

        for ifo in ifos:
            duration = len(background_dict[ifo]) / sample_rate
            psd = normalize_psd(background_dict[ifo], df, sample_rate)
            data = noise_from_psd(psd, df, duration, sample_rate)
            background_dict[ifo] = TimeSeries(data, dt=1 / sample_rate)

    background = np.stack([ts.value for ts in background_dict.values()])
    background = torch.as_tensor(background)

    # instantiate prior, sample, and generate signals
    prior = prior()
    params = prior.sample(n_samples)

    cross, plus = generate_time_domain_sine_gaussian(
        frequencies=params["frequency"],
        hrss=params["hrss"],
        qualities=params["quality"],
        phases=params["phase"],
        eccentricities=params["eccentricity"],
        sample_rate=sample_rate,
        duration=waveform_duration,
    )

    plus = torch.Tensor(plus)
    cross = torch.Tensor(cross)

    kernel_size = int(kernel_length * sample_rate)

    # project raw polarizations onto interferometers
    # with sampled sky localizations
    tensors, vertices = get_ifo_geometry(*ifos)

    # dec is declination
    # psi is polarization angle
    # phi is relative azimuthal angle between source and earth
    dec = torch.Tensor(params["dec"])
    psi = torch.Tensor(params["psi"])

    # The "correct" conversion here doesnt matter
    # since we arent comparing with bilby
    # Just putting phi in the range [-pi, pi]
    # so that it is consistent with the training set
    phi = torch.Tensor(params["ra"]) - np.pi
    params["phi"] = phi

    waveforms = compute_observed_strain(
        dec,
        psi,
        phi,
        tensors,
        vertices,
        sample_rate,
        plus=plus,
        cross=cross,
    )

    # inject signals into randomly sampled kernels of background
    injections = inject_into_random_background(
        background,
        waveforms,
        kernel_size,
    )

    # write signals and parameters used to generate them
    with h5py.File(signal_file, "w") as f:

        f.create_dataset("injections", data=injections)

        for name, value in params.items():
            f.create_dataset(name, data=value)

        # write attributes
        f.attrs.update(
            {
                "size": n_samples,
                "sample_rate": sample_rate,
                "waveform_duration": waveform_duration,
            }
        )

    return signal_file


if __name__ == "__main__":
    main()
