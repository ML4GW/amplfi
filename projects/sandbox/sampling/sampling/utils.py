from pathlib import Path
from typing import Callable, List, Optional, Tuple

import bilby
import h5py
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ml4gw.transforms import ChannelWiseScaler
from mlpe.data.transforms import Preprocessor
from mlpe.injection.utils import phi_from_ra


# TODO: add this function to preprocessor module class
def load_preprocessor_state(
    preprocessor_dir: Path,
    param_dim: int,
    n_ifos: int,
    fduration: float,
    sample_rate: float,
    device: str,
):
    standard_scaler = ChannelWiseScaler(param_dim)
    preprocessor = Preprocessor(
        n_ifos,
        sample_rate,
        fduration,
        scaler=standard_scaler,
    )
    whitener_path = preprocessor_dir / "whitener.pt"
    scaler_path = preprocessor_dir / "scaler.pt"

    preprocessor.whitener = torch.load(whitener_path)
    preprocessor.scaler = torch.load(scaler_path)

    preprocessor = preprocessor.to(device)
    return preprocessor


def initialize_data_loader(
    testing_path: Path,
    inference_params: List[str],
    device: str,
):
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
    injections = torch.from_numpy(signals).to(torch.float32)
    params = torch.from_numpy(params).to(torch.float32)

    dataset = torch.utils.data.TensorDataset(injections, params)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        pin_memory=False if device == "cpu" else True,
        batch_size=1,
        pin_memory_device=device,
    )
    return dataloader, params


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


def load_and_initialize_flow(
    flow: Callable,
    embedding: Callable,
    model_state_path: Path,
    n_ifos: int,
    strain_dim: int,
    param_dim: int,
    device: str,
):
    model_state = torch.load(model_state_path)
    embedding = embedding((n_ifos, strain_dim))
    flow_obj = flow((param_dim, n_ifos, strain_dim), embedding)
    flow_obj.build_flow()
    flow_obj.set_weights_from_state_dict(model_state)
    flow_obj.to_device(device)

    flow = flow_obj.flow
    return flow


def draw_samples_from_model(
    test_dataloader: torch.utils.data.DataLoader,
    flow: torch.nn.Module,
    preprocessor: torch.nn.Module,
    inference_params: List[str],
    num_samples_draw: int,
    priors: dict,
    device: str,
    label: str = "testing_samples",
):
    results = []
    for signal, param in test_dataloader:
        signal = signal.to(device)
        param = param.to(device)
        strain, scaled_param = preprocessor(signal, param)
        with torch.no_grad():
            samples = flow.sample(num_samples_draw, context=strain)
            descaled_samples = preprocessor.scaler(
                samples[0].transpose(1, 0), reverse=True
            )
            descaled_samples = descaled_samples.unsqueeze(0).transpose(2, 1)
            descaled_res = cast_samples_as_bilby_result(
                descaled_samples.cpu().numpy()[0],
                param.cpu().numpy()[0],
                inference_params,
                priors,
                label=label,
            )
        results.append(descaled_res)

    return results


def load_and_sort_bilby_results_from_dynesty(
    bilby_result_dir: Path,
    inference_params: List[str],
    parameters: np.ndarray,
):
    bilby_results = []
    paths = sorted(list(bilby_result_dir.iterdir()))
    for idx, (path, param) in enumerate(zip(paths, parameters)):
        bilby_result = bilby.result.Result.from_pickle(path)
        bilby_result.injection_parameters = {
            k: float(v) for k, v in zip(inference_params, param)
        }
        bilby_result.label = f"bilby_{idx}"
        bilby_results.append(bilby_result)
    return bilby_results


def add_phi_to_bilby_results(results: List[bilby.core.result.Result]):
    """Attach phi w.r.t. GMST to the bilby results"""
    results = []
    for res in results:
        res.posterior["phi"] = phi_from_ra(
            res.posterior["ra"], res.injection_parameters["geocent_time"]
        )
        results.append(res)
    return results
