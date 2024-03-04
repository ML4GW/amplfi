import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import h5py
import numpy as np
import torch
from data_generation.utils import (
    download_data,
    inject_into_background,
    noise_from_psd,
)
from gwpy.timeseries import TimeSeries
from mldatafind.segments import query_segments

from ml4gw.gw import compute_observed_strain, get_ifo_geometry
from ml4gw.spectral import normalize_psd
from mlpe.injection import generate_time_domain_sine_gaussian
from mlpe.injection.utils import phi_from_ra
from mlpe.logging import configure_logging
from typeo import scriptify


@scriptify
def main(
    prior: Callable,
    n_samples: int,
    ifos: List[str],
    state_flag: str,
    frame_type: str,
    channel: str,
    spacing: float,
    start: float,
    stop: float,
    kernel_length: float,
    sample_rate: float,
    waveform_duration: float,
    bilby_duration: float,
    buffer: float,
    datadir: Path,
    logdir: Path,
    waveform_arguments: Optional[Dict] = None,
    parameter_conversion: Optional[Callable] = None,
    gaussian: bool = False,
    verbose: bool = False,
    force_generation: bool = False,
):

    configure_logging(logdir / "generate_testing_set.log", verbose)
    datadir = datadir / "bilby"
    datadir.mkdir(exist_ok=True, parents=True)
    logdir.mkdir(exist_ok=True, parents=True)

    signal_files_exist = all(
        [
            Path(datadir / f"{ifo}_bilby_injections.hdf5").exists()
            for ifo in ifos
        ]
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

    # generate signal times based on requested spacing
    # and save as text file for bilby
    spacing += waveform_duration // 2
    signal_times = np.linspace(
        buffer + waveform_duration // 2, n_samples * spacing, n_samples
    )

    # bilby requests the start time of the signals
    # so subtract half the bilby
    # duration which will enforce that the signal
    # lies in the center of the kernel
    np.savetxt(
        datadir / "start_times.txt", signal_times - (bilby_duration / 2)
    )
    min_duration = (
        signal_times[-1] - signal_times[0] + waveform_duration // 2 + buffer
    )

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
            background_dict[ifo] = TimeSeries(
                data, dt=1 / sample_rate, name=f"{ifo}:{channel}"
            )

    # instantiate prior, sample, and generate signals
    prior = prior()
    params = prior.sample(n_samples)
    params["geocent_time"] = signal_times

    cross, plus = generate_time_domain_sine_gaussian(
        frequencies=params["frequency"],
        hrss=params["hrss"],
        qualities=params["quality"],
        phases=params["phase"],
        eccentricities=params["eccentricity"],
        sample_rate=sample_rate,
        duration=waveform_duration,
    )

    cross = torch.Tensor(cross)
    plus = torch.Tensor(plus)

    # project raw polarizations onto interferometers
    # with sampled sky localizations
    tensors, vertices = get_ifo_geometry(*ifos)

    # dec is declination
    # psi is polarization angle
    # phi is relative azimuthal angle between source and earth

    phi = np.array(
        [
            phi_from_ra(ra, time)
            for ra, time in zip(params["ra"], params["geocent_time"])
        ]
    ).flatten()
    params["phi"] = phi
    dec = torch.Tensor(params["dec"])
    psi = torch.Tensor(params["psi"])
    phi = torch.Tensor(params["phi"])

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

    waveforms = waveforms.numpy()
    timeseries = []
    for i, (ifo, data) in enumerate(background_dict.items()):
        # set start time of data to 0 for simplicity
        data.t0 = 0
        times = data.times.value

        # inject waveforms into background and specified times
        data = inject_into_background(
            (times, data),
            waveforms[:, i, :],
            signal_times,
        )

        data.write(
            datadir / f"{ifo}_timeseries.hdf5",
            format="hdf5",
            overwrite=True,
        )

        timeseries.append(data)

    # save timeseries as numpy array for slicing into
    # format digestible by flow
    timeseries = np.stack(timeseries)

    # save injections as array easily ingestible by flow
    # load in the timeseries data and crop around the injection times
    start_indices = (
        (signal_times - (kernel_length // 2)) * sample_rate
    ).astype("int64")
    end_indices = start_indices + int(kernel_length * sample_rate)

    injections = []
    for start, stop in zip(start_indices, end_indices):
        injections.append(timeseries[:, start:stop])

    injections = np.stack(injections)

    with h5py.File(datadir / "bilby_injections.hdf5", "w") as f:
        f.create_dataset("injections", data=injections)
        for key, value in params.items():
            f.create_dataset(key, data=value)
