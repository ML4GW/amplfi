import sys
from typing import Callable

from mlpe.architectures import architecturize
from mlpe.architectures.wrapper import architectures


def set_argv(*args):
    sys.argv = [None] + list(args)


def test_coupling_flow_wrappers():
    def func(architecture: Callable, learning_rate: float):
        nn = architecture((2, 100))

        # arch will be defined in the dict loop later
        assert isinstance(nn, arch)

        return nn.flow, learning_rate

    wrapped = architecturize(func)

    # now try to use this wrapped function at
    # the "command" line for both architectures
    for name, arch in architectures.items():
        if name not in ("coupling-flow"):
            continue

        set_argv(
            "--learning-rate",
            "1e-3",
            name,
            "--num-flow-steps",
            "3",
            "--num-transform-blocks",
            "5",
        )
        transform, lr = wrapped()

        # make sure the parameters that got passed are correct
        assert lr == 1e-3
