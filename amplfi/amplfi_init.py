#!/usr/bin/env python3

import logging
import os
import shutil
from pathlib import Path
from textwrap import dedent
from typing import Literal, Optional

import yaml
from jsonargparse import ArgumentParser

root = Path(__file__).resolve().parent.parent
data_config = (root / "amplfi" / "data" / "datagen.cfg",)
kubernetes_config = root / "kubernetes" / "train.yaml"
TUNE_CONFIGS = [
    root / "amplfi" / "train" / "configs" / "tune.yaml",
]


TRAIN_CONFIGS = {
    "similarity": [
        root / "amplfi" / "train" / "configs" / "similarity" / "cbc.yaml"
    ],
    "flow": [root / "amplfi" / "train" / "configs" / "flow" / "cbc.yaml"],
}


def fill_kubernetes_template(output: Path, s3_bucket):
    """
    Fill in the kubernetes template with the users environment variables
    """
    with open(kubernetes_config) as f:
        docs = list(yaml.safe_load_all(f))
        config, s3 = docs
        s3["stringData"]["AWS_ACCESS_KEY_ID"] = os.getenv(
            "AWS_ACCESS_KEY_ID", ""
        )
        s3["stringData"]["AWS_SECRET_ACCESS_KEY"] = os.getenv(
            "AWS_SECRET_ACCESS_KEY", ""
        )

        # set remote training config path
        config["spec"]["template"]["spec"]["containers"][0]["args"][2] = (
            f"{s3_bucket}/cbc.yaml"
        )

        # set environment variables that will
        # be used in the training job by lightning
        config["spec"]["template"]["spec"]["containers"][0]["env"][1][
            "value"
        ] = os.getenv("WANDB_API_KEY", "")
        config["spec"]["template"]["spec"]["containers"][0]["env"][2][
            "value"
        ] = s3_bucket
        config["spec"]["template"]["spec"]["containers"][0]["env"][3][
            "value"
        ] = f"{s3_bucket}/data"

        with open(output, "w") as f:
            yaml.safe_dump_all([config, s3], f)


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


def create_remote_runfile(
    path: Path,
    name: str,
    s3_bucket: Path,
):
    rundir = path / name
    config = rundir / "datagen.cfg"
    content = f"""
    #!/bin/bash
    export AMPLFI_DATADIR={s3_bucket}/data

    # launch data generation pipeline
    LAW_CONFIG_FILE={config} law run amplfi.data.DataGeneration --workers 5

    # move config file to remote s3 location
    s3cmd put {rundir}/cbc.yaml {s3_bucket}/cbc.yaml

    # launch job
    kubectl apply -f {rundir}/kubernetes.yaml

    """

    runfile = path / name / "run.sh"
    write_content(content, runfile)


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

    # launch {pipeline} pipeline
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

    parser.add_argument(
        "--remote-train",
        type=bool,
        default=None,
        help="Whether to train remotely on nautilus. "
        "If `True`, will copy a yaml file to the run directory "
        "that contains kubernetes yaml for a remote training job.",
    )

    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=None,
        help="The s3 bucket where training data is stored",
    )
    log_format = "%(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    args = parser.parse_args()
    directory = (
        args.directory.resolve()
        if args.directory
        else Path(os.environ.get("AMPLFI_RUNDIR")).resolve()
    )

    if args.s3_bucket is not None:
        s3_bucket = args.s3_bucket.rstrip("/")
        if not args.s3_bucket.startswith("s3://"):
            raise ValueError(
                "S3 bucket must be in the format s3://{bucket-name}/"
            )
    elif args.remote_train and args.s3_bucket is None:
        raise ValueError(
            "S3 bucket must be provided to train remotely on nautilus"
        )
    else:
        s3_bucket = None

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

    if args.remote_train:
        fill_kubernetes_template(
            directory / args.name / "kubernetes.yaml", s3_bucket
        )
        create_remote_runfile(directory, args.name, s3_bucket)
    else:
        create_runfile(
            directory, args.name, args.mode, args.pipeline, s3_bucket
        )

    logging.info(
        f"Initialized a {args.mode} {args.pipeline} "
        f"pipeline at {directory / args.name}"
    )


if __name__ == "__main__":
    main()
