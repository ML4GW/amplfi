import warnings
from pathlib import Path

import torch
import yaml
from jsonargparse import ArgumentParser
from ml4gw.transforms import ChannelWiseScaler

import amplfi
from amplfi.train.architectures.flows import FlowArchitecture


def from_checkpoint(
    checkpoint_path: Path | str,
    device: str = "cpu",
) -> tuple[FlowArchitecture, ChannelWiseScaler]:
    """
    Load a FlowArchitecture and ChannelWiseScaler from a self-contained
    AMPLFI checkpoint.

    Checkpoints must contain an ``amplfi_config`` key embedded during
    training by ``SaveConfigCallback.on_save_checkpoint``. Older
    checkpoints without this key will raise a ``KeyError``.

    Args:
        checkpoint_path: Path to the Lightning ``.ckpt`` file.
        device: Target device (e.g. ``"cpu"`` or ``"cuda"``).

    Returns:
        ``(arch, scaler)`` — a weight-loaded, eval-mode ``FlowArchitecture``
        and fitted ``ChannelWiseScaler``, both on ``device``.
    """
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )

    if "amplfi_config" not in checkpoint:
        raise KeyError(
            f"Checkpoint '{checkpoint_path}' has no embedded 'amplfi_config'. "
            "This checkpoint was created before self-contained checkpoints "
            "were supported."
        )

    checkpoint_version = checkpoint["amplfi_version"]
    current_version = amplfi.__version__

    if checkpoint_version != current_version:
        warnings.warn(
            f"Checkpoint was saved with AMPLFI v{checkpoint_version} "
            f"but you are loading with v{current_version}. "
            "If loading fails, retry using matching versions.",
            stacklevel=2,
        )

    full_config = yaml.safe_load(checkpoint["amplfi_config"])
    model_init_args = full_config["model"]["init_args"]
    inference_params = full_config["data"]["init_args"]["inference_params"]
    arch_config = model_init_args["arch"]

    arch_config["init_args"]["num_params"] = len(inference_params)
    arch_config["init_args"]["embedding_weights"] = None

    parser = ArgumentParser()
    parser.add_argument("--arch", type=FlowArchitecture)
    args = parser.parse_object({"arch": arch_config})
    args = parser.instantiate_classes(args)

    arch: FlowArchitecture = args.arch

    arch_state_dict = {
        k.removeprefix("model."): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("model.")
    }
    arch.load_state_dict(arch_state_dict)
    arch.eval()
    arch = arch.to(device)

    scaler_state_dict = {
        k.removeprefix("scaler."): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("scaler.")
    }
    scaler = ChannelWiseScaler(len(inference_params))
    scaler.load_state_dict(scaler_state_dict)
    scaler.eval()
    scaler = scaler.to(device)

    return arch, scaler
