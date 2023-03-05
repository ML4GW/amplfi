import os
import subprocess
import sys
from pathlib import Path
from typing import List

from bilby_pipe.job_creation import generate_dag
from bilby_pipe.main import MainInput, write_complete_config_file
from bilby_pipe.parser import create_parser
from bilby_pipe.utils import (
    get_command_line_arguments,
    log_version_information,
    parse_args,
)
from typeo import scriptify

from mlpe.logging import configure_logging


@scriptify
def main(
    datadir: Path,
    writedir: Path,
    logdir: Path,
    ifos: List[str],
    waveform: str,
    accounting_group: str,
    accounting_group_user: str,
    verbose: bool = False,
):

    configure_logging(logdir / "bilby.log", verbose)

    # create a default bilby ini file. AFAIK this is required by bilby.
    # we will overwrite the defaults by passing arguments
    # at the command line below
    default_ini_path = datadir / "bilby" / "default.ini"
    generate_default_ini_args = [
        "bilby_pipe_write_default_ini",
        f"{default_ini_path}",
    ]
    subprocess.run(generate_default_ini_args)

    # construct sys.argv that bilby pipe parser will parse
    data_dict = {
        ifo: str(datadir / "bilby" / f"{ifo}_bilby_injections.h5")
        for ifo in ifos
    }
    psd_dict = {ifo: str(datadir / "psds" / f"{ifo}_psd.txt") for ifo in ifos}
    bilby_outdir = writedir / "bilby" / "rundir"
    bilby_outdir.mkdir(exist_ok=True, parents=True)
    sys.argv = [
        "",
        str(default_ini_path),
        "--accounting",
        accounting_group,
        "--detectors",
        "H1",
        "--detectors",
        "L1",
        "--data-dict",
        str(data_dict),
        "--psd-dict",
        str(psd_dict),
        "--frequency-domain-source-model",
        str(waveform),
        "--gps-file",
        str(datadir / "bilby" / "signal_times.txt"),
        "--outdir",
        str(bilby_outdir),
        "--submit",
    ]

    # necessary due to weird bilby pipe behavior dealing with relative paths
    os.chdir(bilby_outdir)

    # create bilby pipe parser, parse command line args, and launch dag
    parser = create_parser(top_level=True)
    args, unknown_args = parse_args(get_command_line_arguments(), parser)
    log_version_information()
    args.outdir = args.outdir.replace("'", "").replace('"', "")
    inputs = MainInput(args, unknown_args)
    write_complete_config_file(parser, args, inputs)

    print(inputs.outdir, inputs.initialdir)
    generate_dag(inputs)
