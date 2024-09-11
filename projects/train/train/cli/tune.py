from jsonargparse import ActionConfigFile, ArgumentParser
from lightray.tune import run


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        action=ActionConfigFile,
        help="Path to configuration file",
    )

    parser.add_function_arguments(run, skip={"args"})
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
