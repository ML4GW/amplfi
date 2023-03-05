import logging
from pathlib import Path
from typing import List

import h5py
import numpy as np
from data_generation.utils import gaussian_noise_from_gwpy_timeseries
from gwdatafind import find_urls
from gwpy.segments import DataQualityDict
from gwpy.timeseries import TimeSeries
from typeo import scriptify

from mlpe.logging import configure_logging


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
    datadir: Path,
    logdir: Path,
    force_generation: bool = False,
    verbose: bool = False,
    gaussian: bool = False,
):
    """Generates background data for training BBHnet

    Args:
        start: start gpstime
        stop: stop gpstime
        ifos: which ifos to query data for
        outdir: where to store data
    """

    # make logdir dir
    logdir.mkdir(exist_ok=True, parents=True)
    datadir.mkdir(exist_ok=True, parents=True)

    # configure logging output file
    configure_logging(logdir / "generate_background.log", verbose)

    # check if paths already exist
    # TODO: maybe put all background in one path
    background_file = datadir / "background.h5"

    if background_file.exists() and not force_generation:
        logging.info(
            "Background data already exists"
            " and forced generation is off. Not generating background"
        )
        return background_file

    # query segments for each ifo
    # I think a certificate is needed for this
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

        # find frame files
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

        # resample
        data = data.resample(sample_rate)

        if np.isnan(data).any():
            raise ValueError(
                f"The background for ifo {ifo} contains NaN values"
            )

        if gaussian:
            logging.info(f"Generating gaussian noise from psd for ifo {ifo}")
            df = 0.5
            data = gaussian_noise_from_gwpy_timeseries(data, df)

        background_data[ifo] = data

    with h5py.File(background_file, "w") as f:
        for ifo, data in background_data.items():
            f.create_dataset(ifo, data=data)
        f.attrs.update({"t0": float(segment[0])})

    return background_file
