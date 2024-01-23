from typing import Iterable

import torch

from .wrapper import _wrap_callable


# typeo requires type hints to parse arguments from the command line.
# this is a thin wrapper that type hints torch.optim.Adam arguments.
# if there is an argument you wish to specify from the command line,
# you must add it here.
class Adam(torch.optim.Adam):
    def __init__(
        self,
        parameters: Iterable,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        super().__init__(parameters, lr=lr, weight_decay=weight_decay)


optimizers = {"adam": _wrap_callable(Adam)}
