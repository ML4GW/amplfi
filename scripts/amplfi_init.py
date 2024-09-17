#!/usr/bin/env python3

import shutil
from pathlib import Path
from textwrap import dedent
from typing import Literal, Optional

import yaml
from jsonargparse import ArgumentParser

root = Path(__file__).resolve().parent.parent
data_config = (root / "amplfi" / "law" / "datagen.cfg",)
TUNE_CONFIGS = [
    root / "projects" / "train" / "train" / "tune" / "tune.yaml",
    root / "projects" / "train" / "train" / "tune" / "search_space.py",
]


TRAIN_CONFIGS = {
    "similarity": [
        root / "projects" / "train" / "configs" / "similarity" / "cbc.yaml"
    ],
    "flow": [root / "projects" / "train" / "configs" / "flow" / "cbc.yaml"],
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
        if config.name == "tune.yaml":
            with open(config, "r") as f:
                dict = yaml.safe_load(f)
                dict["train_config"] = str(path / "cbc.yaml")

            with open(dest, "w") as f:
                yaml.dump(dict, f)
        else:
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
    mode: Literal["flow", "similarity"],
    pipeline: Literal["tune", "train"],
    s3_bucket: Optional[Path] = None,
):
    # if s3 bucket is provided
    # store training data and training info there
    base = path if s3_bucket is None else s3_bucket

    config = path / "datagen.cfg"
    # make the below one string
    data_cmd = f"LAW_CONFIG_FILE={config} poetry run "
    data_cmd += f"--directory {root / 'amplfi' / 'law'} "
    data_cmd += "law run amplfi.law.DataGeneration --workers 5\n"

    train_root = root / "projects" / "train"
    train_cmd = f"poetry run --directory {train_root} python "

    if pipeline == "tune":
        train_cmd += (
            f"{train_root / 'train' / 'tune' / 'tune.py'} --config tune.yaml"
        )
    else:
        cli = "similarity" if mode == "similarity" else "flow"
        train_cmd += (
            f"{train_root / 'train'  / 'cli' / f'{cli}.py'} "
            "fit --config cbc.yaml"
        )

    content = f"""
    #!/bin/bash
    # Export environment variables
    export AMPLFI_DATADIR={base}
    export AMPLFI_OUTDIR={base}/training/
    export AMPLFI_CONDORDIR={path}/condor

    # launch the data generation pipeline;
    {data_cmd}

    # launch training or tuning pipeline
    {train_cmd}
    """

    runfile = path / "run.sh"
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
        help="Either 'flow' or 'similarity',"
        "Whether to setup a flow or similarity training",
    )
    parser.add_argument(
        "--pipeline",
        choices=["tune", "train"],
        default="train",
        help="Either 'train' or 'tune'."
        "Whether to setup a tune or train pipeline",
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=Path,
        required=True,
        help="The run directory where the"
        "configuration files will be copied to",
    )
    parser.add_argument("--s3-bucket")

    args = parser.parse_args()
    directory = args.directory.resolve()

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

    copy_configs(directory, configs)
    create_runfile(directory, args.mode, args.pipeline, args.s3_bucket)


if __name__ == "__main__":
    main()
