#!/usr/bin/env python3

import logging
import os
import shutil
from pathlib import Path
from textwrap import dedent
from typing import Literal, Optional

from jsonargparse import ArgumentParser

root = Path(__file__).resolve().parent.parent
data_config = (root / "amplfi" / "data" / "datagen.cfg",)
TUNE_CONFIGS = [
    root / "amplfi" / "train" / "configs" / "tune.yaml",
]


TRAIN_CONFIGS = {
    "similarity": [
        root / "amplfi" / "train" / "configs" / "similarity" / "cbc.yaml"
    ],
    "flow": [root / "amplfi" / "train" / "configs" / "flow" / "cbc.yaml"],
}


def copy_configs(
    path: Path,
    configs: list[Path],
):
    """
    Copy the configuration files to the specified directory for editing.

    Any path specific configurations will be updated to point to the
    correct paths in the new directory.

    Args:
        path:
            The directory to copy the configuration files to.
        configs:
            The list of configuration files to copy.
    """

    path.mkdir(parents=True, exist_ok=True)
    for config in configs:
        dest = path / config.name
        shutil.copy(config, dest)


def write_content(content: str, path: Path):
    content = dedent(content).strip("\n")
    with open(path, "w") as f:
        f.write(content)

    # make the file executable
    path.chmod(0o755)
    return content


def create_runfile(
    path: Path,
    name: str,
    mode: Literal["flow", "similarity"],
    pipeline: Literal["tune", "train"],
    s3_bucket: Optional[Path] = None,
):
    # if s3 bucket is provided
    # store training data and training info there
    base = path if s3_bucket is None else s3_bucket

    config = path / name / "datagen.cfg"
    # make the below one string
    data_cmd = f"LAW_CONFIG_FILE={config} "
    data_cmd += "law run amplfi.data.DataGeneration --workers 5"

    if pipeline == "tune":
        train_cmd = "lightray --config tune.yaml -- --config cbc.yaml"
    else:
        train_cmd = f"amplfi-{mode}-cli fit --config cbc.yaml"

    content = f"""
    #!/bin/bash
    # set environment variables for this job
    export AMPLFI_DATADIR={base}/data/
    export AMPLFI_OUTDIR={base}/{name}/
    export AMPLFI_CONDORDIR={path}/data/condor

    # set the GPUs exposed to job
    CUDA_VISIBLE_DEVICES=0

    # launch the data generation pipeline
    {data_cmd}

    # launch {pipeline}ing pipeline
    {train_cmd}
    """

    runfile = path / name / "run.sh"
    write_content(content, runfile)


def main():
    parser = ArgumentParser(
        description="Initialize a directory with configuration files "
        "for running end-to-end amplfi training or tuning pipelines"
    )
    parser.add_argument(
        "--mode",
        choices=["flow", "similarity"],
        default="flow",
        help="Either 'flow' or 'similarity'. "
        "Whether to setup a flow or similarity training",
    )
    parser.add_argument(
        "--pipeline",
        choices=["tune", "train"],
        default="train",
        help="Either 'train' or 'tune'. "
        "Whether to setup a tune or train pipeline",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        required=True,
        help="The name of the run. "
        "This will be used to create the run subdirectory.",
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=Path,
        default=None,
        help="The parent directory where the "
        "data and subdirectories for runs will "
        "be stored. If not provided, the environment "
        "variable AMPLFI_RUNDIR will be used.",
    )

    parser.add_argument("--s3-bucket")
    log_format = "%(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    args = parser.parse_args()
    directory = (
        args.directory.resolve()
        if args.directory
        else Path(os.environ.get("AMPLFI_RUNDIR")).resolve()
    )

    if args.s3_bucket is not None and not args.s3_bucket.startswith("s3://"):
        raise ValueError("S3 bucket must be in the format s3://{bucket-name}/")

    # construct the config files to copy
    # for the given mode and pipeline
    if args.pipeline == "tune":
        configs = TUNE_CONFIGS
        configs.extend(data_config)
        configs.extend(TRAIN_CONFIGS[args.mode])
    else:
        configs = TRAIN_CONFIGS[args.mode]
        configs.extend(data_config)

    copy_configs(directory / args.name, configs)
    create_runfile(
        directory, args.name, args.mode, args.pipeline, args.s3_bucket
    )
    logging.info(
        f"Initialized a {args.mode} {args.pipeline} "
        f"pipeline at {directory / args.name}"
    )


if __name__ == "__main__":
    main()
