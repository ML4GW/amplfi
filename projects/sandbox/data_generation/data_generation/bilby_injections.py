import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from data_generation.utils import (
    download_data,
    gaussian_noise_from_gwpy_timeseries,
    inject_into_background,
)
from gwpy.timeseries import TimeSeries
from mldatafind.segments import query_segments
from typeo import scriptify

from ml4gw.gw import compute_observed_strain, get_ifo_geometry
from mlpe.injection import generate_gw
from mlpe.logging import configure_logging


@scriptify
def main(
    prior: Callable,
    n_samples: int,
    ifos: List[str],
    state_flag: str,
    frame_type: str,
    channel: str,
    waveform: Callable,
    spacing: float,
    start: float,
    stop: float,
    sample_rate: float,
    waveform_duration: float,
    buffer: float,
    datadir: Path,
    logdir: Path,
    waveform_arguments: Optional[Dict] = None,
    parameter_conversion: Optional[Callable] = None,
    gaussian: bool = False,
    verbose: bool = False,
    min_duration: float = 0,
    force_generation: bool = False,
):

    configure_logging(logdir / "generate_testing_set.log", verbose)

    datadir = datadir / "bilby"
    datadir.mkdir(exist_ok=True, parents=True)
    logdir.mkdir(exist_ok=True, parents=True)

    signal_files_exist = all(
        [Path(datadir / f"{ifo}_bilby_injections.h5").exists() for ifo in ifos]
    )

    if signal_files_exist and not force_generation:
        logging.info(
            "Bilby timeseries already exists and forced generation is off. "
            "Not generating bilby timeseries."
        )
        return

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

    if gaussian:
        df = 1 / waveform_duration
        for ifo in ifos:
            background_dict[ifo] = gaussian_noise_from_gwpy_timeseries(
                background_dict[ifo], df
            )

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

    # project raw polarizations onto interferometers
    # with sampled sky localizations
    tensors, vertices = get_ifo_geometry(*ifos)

    # dec is declination
    # psi is polarization angle
    # phi is relative azimuthal angle between source and earth
    dec = torch.Tensor(parameters["dec"])
    psi = torch.Tensor(parameters["psi"])
    phi = torch.Tensor(parameters["phi"])

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

    # generate signal times based on requested spacing
    # and save as text file for bilby
    spacing += waveform_duration // 2
    signal_times = np.linspace(
        buffer + waveform_duration // 2, n_samples * spacing, n_samples
    )
    np.savetxt(datadir / "signal_times.txt", signal_times)
    parameters["geocent_time"] = signal_times

    waveforms = waveforms.numpy()
    for i, (ifo, data) in enumerate(background_dict.items()):
        # set start time of data to 0 for simplicity
        data.t0 = 0
        times = data.times.value
        print(times[0], times[-1])

        # inject waveforms into background and specified times
        data = inject_into_background(
            (times, data),
            waveforms[:, i, :],
            signal_times,
        )

        # package into gwpy timeseries and save as hdf5 files
        data = TimeSeries(
            data, dt=1 / sample_rate, channel=f"{ifo}:{channel}", t0=0
        )
        data.write(
            datadir / f"{ifo}_bilby_injections.hdf5",
            format="hdf5",
            overwrite=True,
        )
