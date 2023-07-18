import logging
import pickle
import random
from pathlib import Path
from time import time
from typing import Callable, List

import bilby
import numpy as np
from bilby.core.prior import Uniform
from sampling.utils import (
    draw_samples_from_model,
    initialize_data_loader,
    load_and_initialize_flow,
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
    num_plot_corner: int,
    verbose: bool = False,
):
    logdir = basedir / "log"
    logdir.mkdir(parents=True, exist_ok=True)
    configure_logging(logdir / "sample_with_flow.log", verbose)
    device = device or "cpu"

    priors = sg_uniform()
    priors["phi"] = Uniform(
        name="phi", minimum=-np.pi, maximum=np.pi, latex_label="phi"
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
    test_dataloader, _, _ = initialize_data_loader(
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

    num_plotted = 0
    for res in results:
        # generate diagnostic posteriors
        if random.random() > 0.5 and num_plotted < num_plot_corner:
            corner_plot_filename = (
                basedir / f"{num_plotted}_descaled_corner.png"
            )
            skymap_filename = basedir / f"{num_plotted}_mollview.png"
            res.plot_corner(
                save=True,
                filename=corner_plot_filename,
                levels=(0.5, 0.9),
            )
            plot_mollview(
                res.posterior["phi"],
                res.posterior["dec"],
                truth=(
                    res.injection_parameters["phi"],
                    res.injection_parameters["dec"],
                ),
                outpath=skymap_filename,
            )
            num_plotted += 1
    logging.info("Total sampling time: {:.1f}(s)".format(total_sampling_time))

    logging.info("Making pp-plot")
    pp_plot_dir = basedir / "pp_plots"
    pp_plot_filename = pp_plot_dir / "pp-plot-test-set-5000.png"
    bilby.result.make_pp_plot(
        results,
        save=True,
        filename=pp_plot_filename,
        keys=inference_params,
    )
    logging.info("PP Plots saved in %s" % (pp_plot_dir))

    logging.info("Saving samples obtained from flow")
    with open(basedir / "flow-samples-as-bilby-result.pickle", "wb") as f:
        pickle.dump(results, f)
