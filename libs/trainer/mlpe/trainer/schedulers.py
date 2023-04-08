from typing import Optional

import torch

from mlpe.trainer import _wrap_callable


# typeo requires type hints to parse arguments from the command line.
# this is a thin wrapper that type hints torch scheduler arguments.
# if there is an argument you wish to specify from the command line,
# you must add it here.
class OneCycleLR(torch.optim.lr_scheduler.OneCycleLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        epochs: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        lr_ramp_epochs: Optional[int] = None,
        anneal_strategy: str = "cos",
    ):
        pct_start = None
        if lr_ramp_epochs is not None:
            pct_start = lr_ramp_epochs / epochs

        super().__init__(
            optimizer,
            max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
        )


schedulers = {"onecycle": _wrap_callable(OneCycleLR)}
