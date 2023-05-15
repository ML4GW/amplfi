import logging
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
from data_generation.utils import noise_from_psd
from gwdatafind import find_urls
from gwpy.frequencyseries import FrequencySeries
from gwpy.segments import DataQualityDict
from gwpy.timeseries import TimeSeries
from mldatafind.authenticate import authenticate

from ml4gw.spectral import normalize_psd
from mlpe.logging import configure_logging
from typeo import scriptify


@scriptify
def main(
    start: float,
    stop: float,
    ifos: List[str],
    sample_rate: float,
    channel: str,
    frame_type: str,
    state_flag: str,
    minimum_length: float,
    df: float,
    datadir: Path,
    logdir: Path,
    gaussian: bool = False,
    psd_file: Optional[Path] = None,
    is_psd: bool = True,
    force_generation: bool = False,
    verbose: bool = False,
):
    """Generates background data onto which injections will
    be made to train the PE model. If `gaussian` is True,
    the requested background segment will be used to calcualte
    a psd from which gaussian noise will be generated. If `psd_file`
    is passed, the psd will be used to generate the gaussian noise.


    Args:
        start:
            start gpstime
        stop:
            stop gpstime
        ifos:
            which ifos to query data for
        sample_rate:
            Frequency at which to sample the data
        channel:
            channel to query data from
        frame_type:
            frame type to query data from
        state_flag:
            state flag that defines good data quality
        minimum_length:
            minimum length of continuous, coincident segment to query
        df:
            Used to calculate psd
        datadir:
            where to store the background data
        logdir:
            where to store the log file
        gaussian:
            whether to generate gaussian noise from the psd
        psd_file:
            path to psd file to use for generating gaussian noise
        is_psd:
            whether the psd file is a psd or an asd
        force_generation:
            Force generation of data
        verbose:
            log verbosely
    """
    authenticate()
    if psd_file is not None and not gaussian:
        raise ValueError(
            "Cannot generate gaussian noise from"
            " requested PSD when gaussian is False"
        )

    # make log and data dirs and configure logging settings
    logdir.mkdir(exist_ok=True, parents=True)
    datadir.mkdir(exist_ok=True, parents=True)
    configure_logging(logdir / "generate_background.log", verbose)

    # make psd dir and check if all necessary data files
    # already exist in the data directory
    psd_dir = datadir / "psds"
    psd_dir.mkdir(exist_ok=True, parents=True)
    psd_files = [Path(f"{psd_dir}/{ifo}_psd.txt") for ifo in ifos]
    psd_files_exist = all([psd_file.exists() for psd_file in psd_files])
    background_file = datadir / "background.h5"

    if background_file.exists() and psd_files_exist and not force_generation:
        logging.info(
            "Background data and psds already exists"
            " and forced generation is off. Not generating background"
        )
        return background_file

    # query segments for each ifo
    # TODO: use mldatafind
    segments = DataQualityDict.query_dqsegdb(
        [f"{ifo}:{state_flag}" for ifo in ifos],
        start,
        stop,
    )

    # create copy of first ifo segment list to start
    intersection = segments[f"{ifos[0]}:{state_flag}"].active.copy()

    # loop over ifos finding segment intersection
    for ifo in ifos:
        intersection &= segments[f"{ifo}:{state_flag}"].active

    # find first continuous segment of minimum length
    segment_lengths = np.array(
        [float(seg[1] - seg[0]) for seg in intersection]
    )
    continuous_segments = np.where(segment_lengths >= minimum_length)[0]

    if len(continuous_segments) == 0:
        raise ValueError(
            "No segments of minimum length, not producing background"
        )

    # choose first of such segments
    segment = intersection[continuous_segments[0]]

    logging.info(
        "Querying coincident, continuous segment "
        "from {} to {}".format(*segment)
    )

    background_data = {}
    for ifo in ifos:

        # find frame files, read in data with gwpy, and resample
        files = find_urls(
            site=ifo.strip("1"),
            frametype=f"{ifo}_{frame_type}",
            gpsstart=start,
            gpsend=stop,
            urltype="file",
        )
        data = TimeSeries.read(
            files,
            channel=f"{ifo}:{channel}",
            start=segment[0],
            end=segment[1],
        )
        data = data.resample(sample_rate)

        if np.isnan(data).any():
            raise ValueError(
                f"The background for ifo {ifo} contains NaN values"
            )

        frequencies = np.arange(0, sample_rate / 2 + df, df)
        if gaussian:
            logging.info(f"Generating gaussian noise from psd for ifo {ifo}")
            if psd_file is not None:
                # if user passed a psd file, load it into a FrequencySeries
                # that normalize_psd can handle
                frequencies, psd = np.loadtxt(psd_file, unpack=True)
                if not is_psd:
                    psd = psd**2

                # ml4gw expects psd to start at 0 Hz, so lets check for that
                # TODO: implement logic that prepends 0 to psd
                # if the passed psd doesn't start at 0
                if frequencies[0] != 0:
                    raise ValueError(
                        "PSD must start at 0 Hz, not {}".format(frequencies[0])
                    )
                data = FrequencySeries(psd, frequencies=frequencies)

            # normalize_psd can handle FrequencySeries or TimeSeries
            psd = normalize_psd(data, df, sample_rate)
            data = noise_from_psd(
                psd, df, len(data) / sample_rate, sample_rate
            )

        else:
            # calculate psd so we can save it for use during bilby analysis
            psd = normalize_psd(data, df, sample_rate)

        # save psd
        np.savetxt(
            psd_dir / f"{ifo}_psd.txt", np.column_stack([frequencies, psd])
        )

        background_data[ifo] = data

    with h5py.File(background_file, "w") as f:
        for ifo, data in background_data.items():
            f.create_dataset(ifo, data=data)
        f.attrs.update({"t0": float(segment[0])})

    return background_file
