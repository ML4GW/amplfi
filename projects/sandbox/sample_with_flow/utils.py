from pathlib import Path
from typing import List

import bilby
import h5py
import numpy as np
import pandas as pd
import torch


# TODO: add this function to preprocessor module class
def load_preprocessor_state(
    preprocessor: torch.nn.Module, preprocessor_dir: Path
):

    whitener_path = preprocessor_dir / "whitener.pt"
    scaler_path = preprocessor_dir / "scaler.pt"

    preprocessor.whitener = torch.load(whitener_path)
    preprocessor.scaler = torch.load(scaler_path)
    return preprocessor


def load_test_data(testing_path: Path, inference_params: List[str]):
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


def cast_samples_as_bilby_result(
    samples: np.ndarray,
    truth: np.ndarray,
    inference_params: List[str],
    priors: bilby.core.prior.Priordict,
):
    """Cast posterior samples as bilby Result object"""
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


def generate_corner_plots(
    results: List[bilby.core.result.Result], writedir: Path
):
    for i, result in enumerate(results):
        filename = writedir / f"corner_{i}.png"
        result.plot_corner(
            save=True,
            filename=filename,
            levels=(0.5, 0.9),
        )
