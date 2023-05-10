import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Tuple

import h5py
import numpy as np
import torch

if TYPE_CHECKING:
    from nflows import transforms


def train_for_one_epoch(
    flow: "transforms.flow",
    optimizer: torch.optim.Optimizer,
    train_dataset: Iterable[Tuple[np.ndarray, np.ndarray]],
    valid_dataset: Iterable[Tuple[np.ndarray, np.ndarray]] = None,
    preprocessor: Optional[torch.nn.Module] = None,
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
    for strain, parameters in train_dataset:
        if preprocessor is not None:
            strain, parameters = preprocessor(strain, parameters)

        optimizer.zero_grad(set_to_none=True)  # reset gradient

        with torch.autocast("cuda", enabled=scaler is not None):
            loss = -flow.log_prob(parameters, context=strain)

        train_loss += loss.detach().sum()
        loss = loss.mean()

        samples_seen += len(parameters)

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
        with torch.no_grad():
            for strain, parameters in valid_dataset:
                strain, parameters = strain.to(device), parameters.to(device)
                if preprocessor is not None:
                    strain, parameters = preprocessor(strain, parameters)
                loss = -flow.log_prob(parameters, context=strain)

                valid_loss += loss.detach().sum()
                loss = loss.mean()
                samples_seen += len(parameters)

        valid_loss /= samples_seen
        msg += f", Valid Loss: {valid_loss:.4e}"
    else:
        valid_loss = None

    current_lr = optimizer.param_groups[0]["lr"]
    msg += f", current LR = {current_lr:.3e}"

    logging.info(msg)
    return train_loss, valid_loss, duration, throughput


def train(
    flow: Callable,
    embedding: Callable,
    optimizer: Callable,
    scheduler: Callable,
    outdir: Path,
    # data params
    train_dataset: Iterable[Tuple[np.ndarray, np.ndarray]],
    valid_dataset: Iterable[Tuple[np.ndarray, np.ndarray]] = None,
    preprocessor: Optional[torch.nn.Module] = None,
    # optimization params
    max_epochs: int = 40,
    init_weights: Optional[Path] = None,
    early_stop: Optional[int] = None,
    # misc params
    device: Optional[str] = None,
    use_amp: bool = False,
    profile: bool = False,
) -> float:
    """Train Flow model on in-memory data
    Args:
        architecture:
            A callable which takes as its only input the number
            of parameters, and dimension of the context and returns
            a nflows.Flow object
        outdir:
            Location to save training artifacts like optimized
            weights, preprocessing objects, and visualizations
        train_dataset:
            An Iterable of (X, y) pairs where X is a batch of training
            data and y is the corresponding targets
        valid_dataset:
            An Iterable of (X, y) pairs where X is a batch of training
            data and y is the corresponding targets
        max_epochs:
            Maximum number of epochs over which to train.
        init_weights:
            Path to weights with which to initialize network. If
            left as `None`, network will be randomly initialized.
            If `init_weights` is a directory, it will be assumed
            that this directory contains a file called `weights.pt`.
        lr:
            Learning rate to use during training.
        early_stop:
            Number of epochs without improvement in validation
            loss before training terminates altogether. Ignored
            if `valid_data is None`.
        device:
            Indicating which device (i.e. cpu or gpu) to run on. Use
            `"cuda"` to use the default GPU available, or `"cuda:{i}`"`,
            where `i` is a valid GPU index on your machine, to specify
            a specific GPU (alternatively, consider setting the environment
            variable `CUDA_VISIBLE_DEVICES=${i}` and using just `"cuda"`
            here).
        profile:
            Whether to generate a tensorboard profile of the
            training step on the first epoch. This will make
            this first epoch slower.
        optimizer_fn:
            Weights optimizer. E.g. ``Adam``,
        optimizer_kwargs:
            Keyword arguments, except ``lr``, to training optimizer. Supply as,
            ``{parameter_key: parameter_value}`` in pyproject.toml,
            e.g. ``{weight_decay: 0}``.
        scheduler_fn:
            Learning rate scheduler. E.g.
            ``torch.optim.lr_scheduler.CosineAnnealingLR``.
        scheduler_kwargs:
            Keyword arguments to scheduler. Supply as,
            ``{parameter_key: parameter_value}`` in project's pyproject.toml
    """

    device = device or "cpu"
    logging.info(f"Device: {device}")
    outdir.mkdir(exist_ok=True)

    # infer the dimension of the parameters
    # and the context from the batch
    strain, parameters = next(iter(train_dataset))
    valid_strain, valid_parameters = next(iter(valid_dataset))
    valid_strain, valid_parameters = valid_strain.to(
        device
    ), valid_parameters.to(device)
    with h5py.File(outdir / "raw_batch.h5", "w") as f:
        f["strain"] = strain.cpu().numpy()
        f["parameters"] = parameters.cpu().numpy()
        f["valid_strain"] = valid_strain.cpu().numpy()
        f["valid_parameters"] = valid_parameters.cpu().numpy()

    if preprocessor is not None:
        strain, parameters = preprocessor(strain, parameters)
        valid_strain, valid_parameters = preprocessor(
            valid_strain, valid_parameters
        )

    with h5py.File(outdir / "batch.h5", "w") as f:
        f["strain"] = strain.cpu().numpy()
        f["parameters"] = parameters.cpu().numpy()
        f["valid_strain"] = valid_strain.cpu().numpy()
        f["valid_parameters"] = valid_parameters.cpu().numpy()

    param_dim = parameters.shape[-1]
    _, n_ifos, strain_dim = strain.shape

    # Creating model, loss function, optimizer and lr scheduler
    logging.info("Building and initializing model")

    # instantiate the embedding network, pass it to the flow
    # object, and then build the flow
    embedding = embedding((n_ifos, strain_dim))
    flow_obj = flow((param_dim, n_ifos, strain_dim), embedding)
    flow_obj.build_flow()
    flow_obj.to_device(device)

    # if we passed a module for preprocessing,
    # include it in the model so that the weights
    # get exported along with everything else
    if preprocessor is not None:
        preprocessor.to(device)

    if init_weights is not None:
        # allow us to easily point to the best weights
        # from another run of this same function
        if init_weights.is_dir():
            init_weights = init_weights / "weights.pt"

        logging.debug(
            f"Initializing model weights from checkpoint '{init_weights}'"
        )
        state_dict = torch.load(init_weights)
        flow_obj.set_weights_from_state_dict(state_dict)

    logging.info(flow_obj.flow)
    logging.info("Initializing loss and optimizer")

    # TODO: Allow different loss functions or optimizers to be passed?
    optimizer = optimizer(flow_obj.flow.parameters())
    lr_scheduler = scheduler(optimizer)

    # start training
    torch.backends.cudnn.benchmark = True

    # start training
    scaler = None
    if use_amp and device.startswith("cuda"):
        scaler = torch.cuda.amp.GradScaler()
    elif use_amp:
        logging.warning("'use_amp' flag set but no cuda device, ignoring")

    best_valid_loss = np.inf
    since_last_improvement = 0
    history = {"train_loss": [], "valid_loss": []}

    logging.info("Beginning training loop")
    for epoch in range(max_epochs):
        if epoch == 0 and profile:
            profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=10),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    outdir / "profile"
                ),
            )
            profiler.start()
        else:
            profiler = None

        logging.info(f"=== Epoch {epoch + 1}/{max_epochs} ===")
        train_loss, valid_loss, duration, throughput = train_for_one_epoch(
            flow_obj.flow,
            optimizer,
            train_dataset,
            valid_dataset,
            preprocessor,
            profiler,
            scaler,
            lr_scheduler,
        )

        history["train_loss"].append(train_loss.cpu().item())

        # do some house cleaning with our
        # validation loss if we have one
        if valid_loss is not None:
            history["valid_loss"].append(valid_loss.cpu().item())

            # update our learning rate scheduler if we
            # indicated a schedule with `patience`
            # if patience is not None:
            #     lr_scheduler.step(valid_loss)

            # save this version of the model weights if
            # we achieved a new best loss, otherwise check
            # to see if we need to early stop based on
            # plateauing validation loss
            if valid_loss < best_valid_loss:
                logging.debug(
                    "Achieved new lowest validation loss, "
                    "saving model weights"
                )
                best_valid_loss = valid_loss

                weights_path = outdir / "weights.pt"
                torch.save(flow_obj.flow.state_dict(), weights_path)
                since_last_improvement = 0

            else:
                if early_stop is not None:
                    since_last_improvement += 1
                    if since_last_improvement >= early_stop:
                        logging.info(
                            "No improvement in validation loss in {} "
                            "epochs, halting training early".format(early_stop)
                        )
                        break

    with h5py.File(outdir / "train_results.h5", "w") as f:
        for key, value in history.items():
            f.create_dataset(key, data=value)

    return history
