import inspect
from typing import Iterable

import torch


# this is a decorator that takes an optimizer and returns a function that
# takes the same arguments as the optimizer, but with the first argument
# (the parameters) removed.
# This is used to wrap the optimizers in
# this file so that they can be used as a callable in the config file.
def _wrap_optimizer(optimizer):
    def func(*args, **kwargs):
        def f(parameters):
            return optimizer(parameters, *args, **kwargs)

        return f

    params = inspect.signature(optimizer).parameters
    params = list(params.values())[1:]
    func.__signature__ = inspect.Signature(params)
    return func


# small wrapper to type hint for typeo;
# if there is a parameter
class Adam(torch.optim.Adam):
    def __init__(
        self,
        parameters: Iterable,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        super().__init__(parameters, lr=lr, weight_decay=weight_decay)


optimizers = {"adam": _wrap_optimizer(Adam)}
