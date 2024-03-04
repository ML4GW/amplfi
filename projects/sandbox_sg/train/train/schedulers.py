from typing import Optional

import torch

from .wrapper import _wrap_callable


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


class ConstantLR(torch.optim.lr_scheduler.ConstantLR):
    def __init__(self, optimizer: torch.optim.Optimizer, total_iters=5):
        super().__init__(optimizer, total_iters=total_iters)


class StepLR(torch.optim.lr_scheduler.StepLR):
    def __init__(
        self, optimizer: torch.optim.Optimizer, step_size: int, gamma: float
    ):
        super().__init__(optimizer, step_size=step_size, gamma=gamma)


class ExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer: torch.optim.Optimizer, gamma=0.99):
        super().__init__(optimizer, gamma=gamma)


class CosineAnnealingLR(torch.optim.lr_scheduler.CosineAnnealingLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        eta_min: float = 1e-5,
    ):
        super().__init__(optimizer, T_max=T_max, eta_min=eta_min)


class SequentialLR(torch.optim.lr_scheduler.SequentialLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        schedulers: list,
        scheduler_kwargs: list,
        milestones: list,
    ):
        """Learning rate schedulers

        Args:
            optimizer:
                Learning rate optimizer object
            schedulers:
                List of string names for schedulers. Assumed to
                be a valid torch scheduler. E.g.
                `['ConstantLR', 'CosineAnnealingLR']` initialized
                with `optimizer`.
            scheduler_kwargs:
                List of keyword arguments passed to ``schedulers``.
                Supply as strings.
                E.g. `["dict(total_iters=5)", "dict(T_max=100)"]`
            milestones:
                List of milestones passed to
                ``torch.optim.lr_scheduler.SequentialLR``.
        """
        _schedulers = [
            getattr(torch.optim.lr_scheduler, str_name)(
                optimizer, **eval(kwargs)
            )
            for str_name, kwargs in zip(schedulers, scheduler_kwargs)
        ]
        super().__init__(
            optimizer,
            schedulers=_schedulers,
            milestones=[eval(_) for _ in milestones],
        )


class ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 0.0001,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: float = 0,
        eps: float = 1e-08,
        verbose: bool = False,
    ):
        super().__init__(
            optimizer,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
            verbose=verbose,
        )


schedulers = {
    "onecycle": _wrap_callable(OneCycleLR),
    "constant": _wrap_callable(ConstantLR),
    "exponential": _wrap_callable(ExponentialLR),
    "cosine": _wrap_callable(CosineAnnealingLR),
    "sequential": _wrap_callable(SequentialLR),
    "plateau": _wrap_callable(ReduceLROnPlateau),
    "step": _wrap_callable(StepLR),
}
