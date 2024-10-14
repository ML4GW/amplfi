import logging
import sys
from typing import Type

from jsonargparse import ActionConfigFile, ArgumentParser
from lightray.tune import run

from ..train.cli.base import AmplfiBaseCLI


def init_logging(verbose: bool):
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        format=log_format,
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stdout,
    )


def main():
    parser = ArgumentParser(parser_mode="omegaconf")
    parser.add_argument(
        "--config",
        action=ActionConfigFile,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    # for now skip args and cli_cls
    parser.add_function_arguments(run, skip={"args", "cli_cls"})
    parser.add_argument("cli_cls", type=Type[AmplfiBaseCLI], help="CLI class")
    parser.add_argument(
        "train_config", type=str, help="Path to training configuration file"
    )

    args = parser.parse_args()
    init_logging(args.pop("verbose"))
    args = parser.instantiate_classes(args)
    train_config = args.pop("train_config")
    args.pop("config")
    train_args = ["--config", train_config]
    run(**vars(args), args=train_args)


if __name__ == "__main__":
    main()
