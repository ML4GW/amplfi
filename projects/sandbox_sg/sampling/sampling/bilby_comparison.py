import logging
from math import pi
from pathlib import Path
from time import time
from typing import Callable, List

import bilby
from bilby.core.prior import Uniform
from sampling.utils import (
    add_phi_to_bilby_results,
    draw_samples_from_model,
    initialize_data_loader,
    load_and_initialize_flow,
    load_and_sort_bilby_results_from_dynesty,
    load_preprocessor_state,
    plot_mollview,
)

from mlpe.architectures import embeddings, flows
from mlpe.injection.priors import sg_uniform
from mlpe.logging import configure_logging
from typeo import scriptify


@scriptify(
    flow=flows,
    embedding=embeddings,
)
def main(
    flow: Callable,
    embedding: Callable,
    model_state_path: Path,
    ifos: List[str],
    sample_rate: float,
    kernel_length: float,
    fduration: float,
    inference_params: List[str],
    basedir: Path,
    testing_set: Path,
    device: str,
    num_samples_draw: int,
    bilby_result_dir: Path = None,
    verbose: bool = False,
):
    logdir = basedir / "log"
    logdir.mkdir(parents=True, exist_ok=True)
    configure_logging(logdir / "sample_with_flow.log", verbose)
    device = device or "cpu"

    priors = sg_uniform()
    priors["phi"] = Uniform(
        name="phi", minimum=-pi, maximum=pi, latex_label="phi"
    )  # FIXME: remove when prior is moved to using torch tools
    n_ifos = len(ifos)
    param_dim = len(inference_params)
    strain_dim = int((kernel_length - fduration) * sample_rate)

    logging.info("Initializing model and setting weights from trained state")
    flow = load_and_initialize_flow(
        flow,
        embedding,
        model_state_path,
        n_ifos,
        strain_dim,
        param_dim,
        device,
    )
    flow.eval()  # set flow to eval mode
    logging.info(
        "Initializing preprocessor and setting weights from trained state"
    )
    preprocessor_dir = basedir / "training" / "preprocessor"
    preprocessor = load_preprocessor_state(
        preprocessor_dir, param_dim, n_ifos, fduration, sample_rate, device
    )
    logging.info("Loading test data and initializing dataloader")
    test_dataloader, params, times = initialize_data_loader(
        testing_set, inference_params, device
    )
    logging.info(
        "Drawing {} samples for each test data".format(num_samples_draw)
    )
    total_sampling_time = time()
    results = draw_samples_from_model(
        test_dataloader,
        flow,
        preprocessor,
        inference_params,
        num_samples_draw,
        priors,
        device,
        label="test_samples_using_flow",
    )
    total_sampling_time = time() - total_sampling_time

    logging.info("Loading bilby results")
    bilby_results = load_and_sort_bilby_results_from_dynesty(
        bilby_result_dir, inference_params, params, times
    )

    bilby_results = add_phi_to_bilby_results(bilby_results)

    skymap_dir = basedir / "skymaps"
    skymap_dir.mkdir(exist_ok=True, parents=True)
    cornerdir = basedir / "corner"
    cornerdir.mkdir(exist_ok=True)
    logging.info("Making joint posterior plots")
    for idx, (flow_res, bilby_res) in enumerate(zip(results, bilby_results)):
        results = [flow_res, bilby_res]
        filename = str(cornerdir / f"corner_{idx}.png")
        bilby.result.plot_multiple(
            results,
            parameters=results[0].injection_parameters,
            save=True,
            filename=filename,
            levels=(0.5, 0.9),
        )

        # TODO: combined skymaps
        plot_mollview(
            flow_res.posterior["phi"].to_numpy().copy(),
            flow_res.posterior["dec"].to_numpy().copy(),
            truth=(
                flow_res.injection_parameters["phi"],
                flow_res.injection_parameters["dec"],
            ),
            outpath=skymap_dir / f"{idx}_mollview_flow.png",
        )
        plot_mollview(
            bilby_res.posterior["phi"].to_numpy().copy(),
            bilby_res.posterior["dec"].to_numpy().copy(),
            truth=(
                bilby_res.injection_parameters["phi"],
                bilby_res.injection_parameters["dec"],
            ),
            outpath=skymap_dir / f"{idx}_mollview_bilby.png",
        )
