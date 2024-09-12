#!/usr/bin/env python3

import shutil
from pathlib import Path
from textwrap import dedent
from typing import Optional

from jsonargparse import ArgumentParser

root = Path(__file__).resolve().parent.parent
data_config = (root / "amplfi" / "law" / "datagen.cfg",)
TUNE_CONFIGS = [
    root / "projects" / "train" / "tune" / "tune.yaml",
    root / "projects" / "train" / "tune" / "search_space.py",
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
        shutil.copy(config, dest)


def write_content(content: str, path: Path):
    content = dedent(content).strip("\n")
    with open(path, "w") as f:
        f.write(content)

    # make the file executable
    path.chmod(0o755)
    return content


def create_runfile(path: Path, mode: str, s3_bucket: Optional[Path] = None):
    # if s3 bucket is provided
    # store training data and training info there
    base = path if s3_bucket is None else s3_bucket

    config = path / "datagen.cfg"
    # make the below one string
    data_cmd = f"LAW_CONFIG_FILE={config} poetry run "
    data_cmd += f"--directory {root / 'amplfi' / 'law'} "
    data_cmd += "law run amplfi.DataGeneration --workers 5\n"

    train_root = root / "projects" / "train"
    train_cmd = f"poetry run --directory {train_root} python "
    cli = "similarity" if mode == "similarity" else "flow"
    train_cmd += (
        f"{train_root / 'train'  / 'cli' / f'{cli}.py'} fit --config {config}"
    )

    content = f"""
    #!/bin/bash
    # Export environment variables
    export AMPLFI_DATADIR={base}
    export AMPLFI_OUTDIR={base}/training/
    export AMPLFI_CONDORDIR={path}/condor

    # launch pipeline; modify the gpus, workers etc. to suit your needs
    # note that if you've made local code changes not in the containers
    # you'll need to add the --dev flag!
    {data_cmd}
    {train_cmd}
    """

    runfile = path / "run.sh"
    write_content(content, runfile)


def main():
    # offline subcommand (sandbox or tune)
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["flow", "similarity"],
        default="flow",
        help="Whether to setup a flow or similarity training",
    )
    parser.add_argument(
        "--pipeline",
        choices=["tune", "train"],
        default="train",
        help="Whether to setup a tune or train pipeline",
    )
    parser.add_argument("-d", "--directory", type=Path, required=True)
    parser.add_argument("--s3-bucket")

    args = parser.parse_args()
    directory = args.directory.resolve()

    if args.s3_bucket is not None and not args.s3_bucket.startswith("s3://"):
        raise ValueError("S3 bucket must be in the format s3://{bucket-name}/")

    # construct the config files to copy
    # for the given mode and pipeline
    if args.mode == "tune":
        configs = TUNE_CONFIGS
        configs.extend(data_config)
        configs.extend(TRAIN_CONFIGS[args.mode])
    else:
        configs = TRAIN_CONFIGS[args.mode]
        configs.extend(data_config)

    copy_configs(directory, configs)
    create_runfile(directory, args.mode, args.s3_bucket)


if __name__ == "__main__":
    main()
