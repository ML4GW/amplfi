import logging
import random
from pathlib import Path
from time import time
from typing import List

import bilby
import h5py
import numpy as np
import pandas as pd
import torch

from ml4gw.transforms import ChannelWiseScaler
from mlpe.architectures import architecturize
from mlpe.data.transforms import Preprocessor
from mlpe.injection.priors import sg_uniform
from mlpe.logging import configure_logging


def _load_preprocessor_state(preprocessor, preprocessor_dir):
    preprocessor_path = Path(preprocessor_dir) / "preprocessor"

    whitener_path = preprocessor_path / "whitener.pt"
    scaler_path = preprocessor_path / "scaler.pt"

    preprocessor.whitener = torch.load(whitener_path)
    preprocessor.scaler = torch.load(scaler_path)
    return preprocessor


def _load_test_data(testing_path: Path, inference_params: List[str]):
    with h5py.File(testing_path, "r") as f:
        signals = f["injections"][:]
        params = []
        for param in inference_params:
            values = f[param][:]
            # take logarithm since hrss
            # spans large magnitude range
            if param == "hrss":
                values = np.log10(values)
            params.append(values)

        params = np.vstack(params).T
    return signals, params


def _cast_as_bilby_result(samples, truth, inference_params, priors):
    """Cast samples as bilby Result object"""
    # samples shape (1, num_samples, num_params)
    # inference_params shape (1, num_params)
    samples = samples[0]
    truth = truth[0]
    injections = {k: float(v) for k, v in zip(inference_params, truth)}

    posterior = dict()
    for idx, k in enumerate(inference_params):
        posterior[k] = samples.T[idx].flatten()
    posterior = pd.DataFrame(posterior)
    return bilby.result.Result(
        label="test_data",
        injection_parameters=injections,
        posterior=posterior,
        search_parameter_keys=inference_params,
        priors=priors,
    )


@architecturize
def main(
    architecture: callable,
    model_state_path: Path,
    ifos: List[str],
    sample_rate: float,
    kernel_length: float,
    fduration: float,
    inference_params: List[str],
    datadir: Path,
    logdir: Path,
    outdir: Path,
    device: str,
    num_samples_draw: int,
    num_plot_corner: int,
    verbose: bool = False,
):
    device = device or "cpu"
    priors = sg_uniform()

    configure_logging(logdir / "sample.log", verbose)
    num_ifos = len(ifos)
    num_params = len(inference_params)
    signal_length = int((kernel_length - fduration) * sample_rate)

    logging.info("Initializing model and setting weights from trained state")
    model_state = torch.load(model_state_path)
    flow_obj = architecture((num_params, num_ifos, signal_length))
    flow_obj.build_flow()
    flow_obj.to_device(device)
    flow_obj.set_weights_from_state_dict(model_state)

    logging.info(
        "Initializing preprocessor and setting weights from trained state"
    )
    standard_scaler = ChannelWiseScaler(num_params)
    preprocessor = Preprocessor(
        num_ifos,
        sample_rate,
        fduration,
        scaler=standard_scaler,
    )

    preprocessor = _load_preprocessor_state(preprocessor, outdir)
    preprocessor = preprocessor.to(device)

    logging.info("Loading test data and initializing dataloader")
    test_data, test_params = _load_test_data(
        datadir / "test_injections.h5", inference_params
    )
    test_data = torch.from_numpy(test_data).to(torch.float32)
    test_params = torch.from_numpy(test_params).to(torch.float32)

    test_dataset = torch.utils.data.TensorDataset(test_data, test_params)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        pin_memory=False if device == "cpu" else True,
        batch_size=1,
        pin_memory_device=device,
    )

    logging.info(
        "Drawing {} samples for each test data".format(num_samples_draw)
    )
    results = []
    descaled_results = []
    total_sampling_time = 0.0
    num_plotted = 0  # corner plots for diagnostic analysis
    for signal, param in test_dataloader:
        signal = signal.to(device)
        param = param.to(device)
        strain, scaled_param = preprocessor(signal, param)

        _time = time()
        with torch.no_grad():
            samples = flow_obj.flow.sample(num_samples_draw, context=strain)
            descaled_samples = preprocessor.scaler(
                samples[0].transpose(1, 0), reverse=True
            )
            descaled_samples = descaled_samples.unsqueeze(0).transpose(2, 1)
        _time = time() - _time

        descaled_res = _cast_as_bilby_result(
            descaled_samples.cpu().numpy(),
            param.cpu().numpy(),
            inference_params,
            priors,
        )
        descaled_results.append(descaled_res)

        res = _cast_as_bilby_result(
            samples.cpu().numpy(),
            scaled_param.cpu().numpy(),
            inference_params,
            priors,
        )
        results.append(res)

        logging.debug("Time taken to sample: %.2f" % (_time))
        total_sampling_time += _time

        # generate diagnostic posteriors
        if random.random() > 0.5 and num_plotted < num_plot_corner:
            scaled_corner_plot_filename = (
                outdir / f"{num_plotted}_scaled_corner.png"
            )
            res.plot_corner(
                save=True,
                filename=scaled_corner_plot_filename,
                levels=(0.5, 0.9),
            )

            descaled_corner_plot_filename = (
                outdir / f"{num_plotted}_descaled_corner.png"
            )
            descaled_res.plot_corner(
                save=True,
                filename=descaled_corner_plot_filename,
                levels=(0.5, 0.9),
            )
            num_plotted += 1

    logging.info(
        "Total/Avg. samlping time: {:.1f}/{:.2f}(s)".format(
            total_sampling_time, total_sampling_time / num_samples_draw
        )
    )
    logging.info("Making pp-plot")
    pp_plot_scaled_filename = outdir / "pp-plot-test-set-scaled.png"
    pp_plot_filename = outdir / "pp-plot-test-set.png"

    bilby.result.make_pp_plot(
        results,
        save=True,
        filename=pp_plot_scaled_filename,
        keys=inference_params,
    )

    bilby.result.make_pp_plot(
        descaled_results,
        save=True,
        filename=pp_plot_filename,
        keys=inference_params,
    )
    logging.info("Plot saved in %s" % (pp_plot_filename))
