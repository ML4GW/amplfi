import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, List

from bilby_pipe.job_creation import generate_dag
from bilby_pipe.main import MainInput, write_complete_config_file
from bilby_pipe.parser import create_parser
from bilby_pipe.utils import log_version_information, parse_args

from mlpe.logging import configure_logging
from typeo import scriptify


@scriptify
def main(
    datadir: Path,
    logdir: Path,
    channel: str,
    ifos: List[str],
    waveform: str,
    prior: Callable,
    bilby_duration: float,
    accounting_group: str,
    sample_rate: float,
    request_cpus: int = 1,
    n_live: int = 1000,
    n_act: int = 5,
    submit: bool = True,
    verbose: bool = False,
    force_generation: bool = False,
):
    """
    Launch a bilby_pipe run on a set of injections
    """

    configure_logging(logdir / "bilby.log", verbose)
    bilby_outdir = datadir / "bilby" / "rundir"
    # TODO: check that the results dir contains all of the results
    if bilby_outdir.exists() and not force_generation:
        logging.info("Bilby output directory already exists. Skipping.")
        return

    bilby_outdir.mkdir(exist_ok=True, parents=True)
    prior = prior()

    # create a default bilby ini file. AFAIK this is required by bilby.
    # we will overwrite the defaults by passing arguments via "args.argument"
    default_ini_path = datadir / "bilby" / "default.ini"
    os.environ["PATH"] = ":".join(sys.path)
    generate_default_ini_args = [
        shutil.which("bilby_pipe_write_default_ini"),
        f"{default_ini_path}",
    ]
    subprocess.run(generate_default_ini_args)

    # load in default bilby inputs from ini path
    parser = create_parser(top_level=True)
    args, _ = parse_args([str(default_ini_path)], parser)

    # construct sys.argv that bilby pipe parser will parse
    data_dict = {
        ifo: str(datadir / "bilby" / f"{ifo}_timeseries.hdf5") for ifo in ifos
    }

    # channels are named the same as the ifos
    channel_dict = {ifo: channel for ifo in ifos}
    psd_dict = {ifo: str(datadir / "psds" / f"{ifo}_psd.txt") for ifo in ifos}
    sampler_kwargs = {
        "sample": "rwalk",
        "nlive": n_live,
        "nact": n_act,
        "naccept": n_act,
        "check_point_plot": True,
        "check_point_delta_t": 1800,
        "print_method": "interval-60",
    }

    args.overwrite_outdir = True
    args.plot_data = True
    args.plot_trace = True
    args.plot_corner = False  # getting errors with this
    args.plot_skymap = True
    # args.result_format = "pickle"
    args.duration = bilby_duration
    args.request_cpus = request_cpus
    args.enforce_signal_duration = False
    args.default_prior = "PriorDict"
    args.prior_dict = str(prior).replace("'", "")
    args.data_dict = str(data_dict).replace("'", "")
    args.psd_dict = str(psd_dict).replace("'", "")
    args.sampler_kwargs = str(sampler_kwargs).replace("'", "")
    args.channel_dict = str(channel_dict).replace("'", "")
    args.accounting = accounting_group
    args.detectors = ifos
    args.frequency_domain_source_model = waveform
    args.gps_file = str(datadir / "bilby" / "start_times.txt")
    args.outdir = str(bilby_outdir)
    args.submit = submit
    args.sampling_frequency = sample_rate
    args.waveform_generator = "bilby.gw.waveform_generator.WaveformGenerator"
    args.conversion_function = "noconvert"

    # necessary due to weird bilby pipe behavior dealing with relative paths
    os.chdir(datadir / "bilby")

    # launch dag
    log_version_information()
    inputs = MainInput(args, [])
    write_complete_config_file(parser, args, inputs)
    generate_dag(inputs)


if __name__ == "__main__":
    main()
