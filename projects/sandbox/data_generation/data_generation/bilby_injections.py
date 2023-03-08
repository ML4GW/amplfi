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
from typeo import scriptify

from ml4gw.gw import compute_observed_strain, get_ifo_geometry
from ml4gw.spectral import normalize_psd
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
        df = 1 / waveform_duration

        for ifo in ifos:
            duration = len(background_dict[ifo]) / sample_rate
            psd = normalize_psd(background_dict[ifo], df, sample_rate)
            data = noise_from_psd(psd, df, duration, sample_rate)
            background_dict[ifo] = TimeSeries(data, dt=1 / sample_rate)

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
    phi = torch.Tensor(parameters["ra"]) - np.pi

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

    parameters["geocent_time"] = signal_times

    waveforms = waveforms.numpy()
    for i, (ifo, data) in enumerate(background_dict.items()):
        # set start time of data to 0 for simplicity
        data.t0 = 0
        times = data.times.value

        # inject waveforms into background and specified times
        print(signal_times, times[-1])
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

    # save parameters as hdf5 file
    with h5py.File(datadir / "bilby_injection_parameters.hdf5", "w") as f:
        for key, value in parameters.items():
            f.create_dataset(key, data=value)