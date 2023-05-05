from pathlib import Path
from typing import List, Optional, Tuple

import bilby
import h5py
import healpy as hp
import matplotlib.pyplot as plt
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
    priors: "bilby.core.prior.Priordict",
    label: str,
):
    """Cast posterior samples as bilby Result object"""
    # samples shape (1, num_samples, num_params)
    # inference_params shape (1, num_params)

    injections = {k: float(v) for k, v in zip(inference_params, truth)}

    posterior = dict()
    for idx, k in enumerate(inference_params):
        posterior[k] = samples.T[idx].flatten()
    posterior = pd.DataFrame(posterior)

    return bilby.result.Result(
        label=label,
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
            parameters=result.injection_parameters,
            save=True,
            filename=filename,
            levels=(0.5, 0.9),
        )


def generate_overlapping_corner_plots(
    results: List[Tuple[bilby.core.result.Result]], writedir: Path
):
    for i, result in enumerate(results):
        filename = writedir / f"corner_{i}.png"
        bilby.result.plot_multiple(
            result,
            parameters=["ra", "dec", "psi"],
            save=True,
            filename=filename,
            levels=(0.5, 0.9),
        )


def plot_mollview(
    ra_samples: np.ndarray,
    dec_samples: np.ndarray,
    nside: int = 32,
    truth: Optional[Tuple[float, float]] = None,
    outpath: Path = None,
):
    """Plot mollview of posterior samples

    Args:
        ra_samples: array of right ascension samples in radians (0, 2 pi)
        dec_samples: array of declination samples in radians (-pi/2, pi/2)
        nside: nside parameter for healpy
        truth: tuple of true ra and dec
    """

    # mask out non physical samples;
    # convert dec between 0 and pi in rads as required by healpy
    ra_samples_mask = (ra_samples > 0) * (ra_samples < 360)
    dec_samples += np.pi / 2
    dec_samples_mask = (dec_samples > 0) * (dec_samples < np.pi)

    net_mask = ra_samples_mask * dec_samples_mask
    ra_samples = ra_samples[net_mask]
    dec_samples = dec_samples[net_mask]

    # calculate number of samples in each pixel
    NPIX = hp.nside2npix(nside)
    ipix = hp.ang2pix(nside, dec_samples, ra_samples)
    ipix = np.sort(ipix)
    uniq, counts = np.unique(ipix, return_counts=True)

    # create empty map and then fill in non-zero pix with counts
    m = np.zeros(NPIX)
    m[np.in1d(range(NPIX), uniq)] = counts

    plt.close()
    # plot molleweide
    fig = hp.mollview(m)
    if truth is not None:
        ra_inj, dec_inj = truth
        dec_inj += np.pi / 2
        hp.visufunc.projscatter(
            dec_inj, ra_inj, marker="x", color="red", s=150
        )

    plt.savefig(outpath)

    return fig
