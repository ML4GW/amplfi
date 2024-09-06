from jsonargparse import ActionConfigFile, ArgumentParser
from lightray import run


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_function_arguments(run)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
