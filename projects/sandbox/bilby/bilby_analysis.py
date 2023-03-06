import os
import subprocess
from pathlib import Path
from typing import List

from bilby_pipe.job_creation import generate_dag
from bilby_pipe.main import MainInput, write_complete_config_file
from bilby_pipe.parser import create_parser
from bilby_pipe.utils import log_version_information, parse_args
from typeo import scriptify

from mlpe.logging import configure_logging


@scriptify
def main(
    datadir: Path,
    writedir: Path,
    logdir: Path,
    channel: str,
    ifos: List[str],
    waveform: str,
    accounting_group: str,
    accounting_group_user: str,
    sample_rate: float,
    verbose: bool = False,
):

    configure_logging(logdir / "bilby.log", verbose)
    bilby_outdir = writedir / "bilby" / "rundir"
    bilby_outdir.mkdir(exist_ok=True, parents=True)

    # create a default bilby ini file. AFAIK this is required by bilby.
    # we will overwrite the defaults by passing arguments
    # at the command line below
    default_ini_path = datadir / "bilby" / "default.ini"
    generate_default_ini_args = [
        "bilby_pipe_write_default_ini",
        f"{default_ini_path}",
    ]
    subprocess.run(generate_default_ini_args)

    # load in default bilby inputs from ini path
    parser = create_parser(top_level=True)
    args, _ = parse_args([str(default_ini_path)], parser)

    # construct sys.argv that bilby pipe parser will parse
    data_dict = {
        ifo: str(datadir / "bilby" / f"{ifo}_bilby_injections.hdf5")
        for ifo in ifos
    }

    # channels are named the same as the ifos
    channel_dict = {ifo: channel for ifo in ifos}
    psd_dict = {ifo: str(datadir / "psds" / f"{ifo}_psd.txt") for ifo in ifos}

    args.data_dict = str(data_dict).replace("'", "")
    args.psd_dict = str(psd_dict).replace("'", "")
    args.channel_dict = str(channel_dict).replace("'", "")
    args.accounting = accounting_group
    args.detectors = ifos
    args.frequency_domain_source_model = waveform
    args.gps_file = str(datadir / "bilby" / "signal_times.txt")
    args.outdir = str(bilby_outdir)
    args.submit = True
    args.sampling_frequency = sample_rate
    args.waveform_generator = "bilby.gw.waveform_generator.WaveformGenerator"

    # necessary due to weird bilby pipe behavior dealing with relative paths
    os.chdir(writedir / "bilby")

    # launch dag
    log_version_information()
    inputs = MainInput(args, [])
    write_complete_config_file(parser, args, inputs)
    generate_dag(inputs)
