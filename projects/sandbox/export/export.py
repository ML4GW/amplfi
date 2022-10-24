import logging
import shutil
from pathlib import Path
from typing import Callable, Optional

import h5py  # noqa
import torch
from mlpe.architectures import architecturize
from mlpe.data.transforms import Preprocessor, StandardScalerTransform
from mlpe.logging import configure_logging

import hermes.quiver as qv


def scale_model(model, instances):
    # TODO: should quiver handle this under the hood?
    try:
        model.config.scale_instance_group(instances)
    except ValueError:
        model.config.add_instance_group(count=instances)


@architecturize
def export(
    architecture: Callable,
    repository_directory: str,
    outdir: Path,
    num_ifos: int,
    kernel_length: float,
    inference_sampling_rate: float,
    sample_rate: float,
    batch_size: int,
    fduration: Optional[float] = None,
    highpass: Optional[float] = None,
    weights: Optional[Path] = None,
    instances: Optional[int] = None,
    platform: qv.Platform = qv.Platform.ONNX,
    clean: bool = False,
    verbose: bool = False,
) -> None:
    """
    Export a PE architecture to a model repository
    for inference
    Args:
        architecture:
            A function which takes as input a number of witness
            channels and returns an instantiated torch `Module`
            which represents a DeepClean network architecture
        repository_directory:
            Directory to which to save the models and their
            configs
        outdir:
            Path to save logs. If `weights` is `None`, this
            directory is assumed to contain a file `"weights.pt"`.
        num_ifos:
            The number of interferometers contained along the
            channel dimension used to train BBHNet
        kernel_length:
            The length, in seconds, of the input to DeepClean
        sample_rate:
            Rate at which the input kernel has been sampled, in Hz
        weights:
            Path to a set of trained weights with which to
            initialize the network architecture. If left as
            `None`, a file called `"weights.pt"` will be looked
            for in the `output_directory`.
        instances:
            The number of concurrent execution instances of the
            mlpe architecture to host per GPU during inference
        platform:
            The backend framework platform used to host the
            DeepClean architecture on the inference service. Right
            now only `"onnxruntime_onnx"` is supported.
        clean:
            Whether to clear the repository directory before starting
            export
        verbose:
            If set, log at `DEBUG` verbosity, otherwise log at
            `INFO` verbosity.
        **kwargs:
            key word arguments specific to the export platform
    """

    # make relevant directories
    logging.info(architecture)
    outdir.mkdir(exist_ok=True, parents=True)

    # if we didn't specify a weights filename, assume
    # that a "weights.pt" lives in our output directory
    if weights is None or weights.is_dir():
        weights_dir = outdir if weights is None else weights
        weights = weights_dir / "weights.pt"
    if not weights.exists():
        raise FileNotFoundError(f"No weights file '{weights}'")

    configure_logging(outdir / "export.log", verbose)

    # instantiate the architecture and initialize
    # its weights with the trained values
    logging.info(f"Creating model and loading weights from {weights}")

    # TODO how to infer param and context dim
    nn = architecture()
    preprocessor = Preprocessor(
        num_ifos,
        sample_rate,
        kernel_length,
        normalizer=StandardScalerTransform,
        fduration=fduration,
        highpass=highpass,
    )
    nn = torch.nn.Sequential(preprocessor, nn)
    nn.load_state_dict(torch.load(weights))
    nn.eval()

    # instantiate a model repository at the
    # indicated location and see if a bbhnet
    # model already exists in this repository
    if clean:
        shutil.rmtree(repository_directory)
    repo = qv.ModelRepository(repository_directory)
    try:
        mlpe = repo.models["mlpe"]
    except KeyError:
        mlpe = repo.add("mlpe", platform=platform)

    # if we specified a number of bbhnet instances
    # we want per-gpu at inference time, scale it now
    if instances is not None:
        scale_model(mlpe, instances)

    # export this version of the model (with its current
    # weights), to this entry in the model repository
    input_shape = (batch_size, num_ifos, int(kernel_length * sample_rate))

    # TODO: hardcoding these kwargs for now, but worth
    # thinking about a more robust way to handle this
    kwargs = {}
    if platform == qv.Platform.ONNX:
        kwargs["opset_version"] = 13

        # turn off graph optimization because of this error
        # https://github.com/triton-inference-server/server/issues/3418
        mlpe.config.optimization.graph.level = -1
    elif platform == qv.Platform.TENSORRT:
        kwargs["use_fp16"] = True

    mlpe.export_version(
        nn,
        input_shapes={"hoft": input_shape},
        output_names=["discriminator"],
        **kwargs,
    )


if __name__ == "__main__":
    export()
