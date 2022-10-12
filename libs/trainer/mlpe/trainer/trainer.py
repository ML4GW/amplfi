import logging
import time
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Tuple

import numpy as np
import torch

if TYPE_CHECKING:
    from nflows import transforms


def train_for_one_epoch(
    flow: "transforms.flow",
    optimizer: torch.optim.Optimizer,
    train_dataset: Iterable[Tuple[np.ndarray, np.ndarray]],
    valid_dataset: Iterable[Tuple[np.ndarray, np.ndarray]] = None,
    profiler: Optional[torch.profiler.profile] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    scheduler: Optional[Callable] = None,
):
    """Run a single epoch of training"""

    train_loss = 0
    samples_seen = 0
    start_time = time.time()
    flow.train()
    device = next(flow.parameters()).device

    for data, context in train_dataset:
        optimizer.zero_grad(set_to_none=True)  # reset gradient

        with torch.autocast("cuda", enabled=scaler is not None):
            loss = -flow.log_prob(data, context=context)

        train_loss += loss.detach().sum()
        loss = loss.mean()
        samples_seen += len(data)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if profiler is not None:
            profiler.step()
        if scheduler is not None:
            scheduler.step()

    if profiler is not None:
        profiler.stop()

    end_time = time.time()
    duration = end_time - start_time
    throughput = samples_seen / duration
    train_loss /= samples_seen

    logging.info(
        "Duration {:0.2f}s, Throughput {:0.1f} samples/s".format(
            duration, throughput
        )
    )
    msg = f"Train Loss: {train_loss:.4e}"

    # Evaluate performance on validation set if given
    if valid_dataset is not None:
        valid_loss = 0
        samples_seen = 0

        flow.eval()

        # reason mixed precision is not used here?
        # since no gradient calculation that requires
        # higher precision?
        with torch.no_grad():
            for data, context in valid_dataset:
                data, context = data.to(device), context.to(device)
                loss = -flow.log_prob(data, context=context)

                valid_loss += loss.detach().sum()
                loss = loss.mean()
                samples_seen += len(data)

        valid_loss /= samples_seen
        msg += f", Valid Loss: {valid_loss:.4e}"
    else:
        valid_loss = None

    logging.info(msg)
    return train_loss, valid_loss, duration, throughput
