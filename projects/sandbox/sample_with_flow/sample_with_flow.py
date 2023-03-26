import logging
from pathlib import Path
from time import time
from typing import List

import bilby
import h5py
import numpy as np
import torch
from bilby.core.prior import Uniform
from gwpy.timeseries import TimeSeries
from utils import (
    cast_samples_as_bilby_result,
    generate_corner_plots,
    load_preprocessor_state,
    load_test_data,
)

from ml4gw.transforms import ChannelWiseScaler
from mlpe.architectures import architecturize
from mlpe.data.transforms import Preprocessor
from mlpe.injection.priors import sg_uniform
from mlpe.injection.utils import ra_from_phi
from mlpe.logging import configure_logging


@architecturize
def main(
    architecture: callable,
    model_state_path: Path,
    ifos: List[str],
    channel: str,
    sample_rate: float,
    kernel_length: float,
    fduration: float,
    inference_params: List[str],
    preprocessor_dir: Path,
    datadir: Path,
    logdir: Path,
    writedir: Path,
    basedir: Path,
    device: str,
    num_samples_draw: int,
    verbose: bool = False,
):
    device = device or "cpu"

    configure_logging(logdir / "pp_plot.log", verbose)

    priors = sg_uniform()
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

    preprocessor = load_preprocessor_state(preprocessor, preprocessor_dir)
    preprocessor = preprocessor.to(device)

    logging.info("Loading test data and initializing dataloader")
    test_data, test_params = load_test_data(
        datadir / "pp_plot_injections.h5", inference_params
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

    priors["phi"] = Uniform(
        name="phi", minimum=0, maximum=2 * np.pi, latex_label="phi"
    )
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

        descaled_res = cast_samples_as_bilby_result(
            descaled_samples.cpu().numpy()[0],
            param.cpu().numpy()[0],
            inference_params,
            priors,
            label="flow",
        )
        descaled_results.append(descaled_res)

        res = cast_samples_as_bilby_result(
            samples.cpu().numpy()[0],
            scaled_param.cpu().numpy()[0],
            inference_params,
            priors,
            label="flow",
        )
        results.append(res)

        logging.debug("Time taken to sample: %.2f" % (_time))
        total_sampling_time += _time

    logging.info(
        "Total/Avg. samlping time: {:.1f}/{:.2f}(s)".format(
            total_sampling_time, total_sampling_time / num_samples_draw
        )
    )

    logging.info("Making pp-plot")
    pp_plot_dir = writedir / "pp_plots"
    pp_plot_scaled_filename = pp_plot_dir / "pp-plot-test-set-scaled.png"
    pp_plot_filename = pp_plot_dir / "pp-plot-test-set.png"

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
    logging.info("PP Plots saved in %s" % (pp_plot_dir))

    # TODO: this project is getting long. Should we split it up?
    # What's the best way to do this?

    # now analyze the injections we've analyzed with bilby.
    logging.info("Comparing with bilby results")

    # load in the bilby injection parameters
    params = []
    with h5py.File(datadir / "bilby" / "bilby_injection_parameters.hdf5") as f:
        times = np.array(f["geocent_time"][:])
        for param in inference_params:
            values = f[param][:]
            # take logarithm since hrss
            # spans large magnitude range
            if param == "hrss":
                values = np.log10(values)
            params.append(values)

        params = np.vstack(params).T

    # load in the timeseries data and crop around the injection times
    timeseries = []
    injections = []

    for ifo in ifos:
        data = TimeSeries.read(
            datadir / "bilby" / f"{ifo}_timeseries.hdf5", format="hdf5"
        )
        timeseries.append(data.value)

    timeseries = np.stack(timeseries)

    start_indices = ((times - (kernel_length // 2)) * sample_rate).astype(
        "int64"
    )
    end_indices = start_indices + int(kernel_length * sample_rate)

    for start, stop in zip(start_indices, end_indices):
        injections.append(timeseries[:, start:stop])

    injections = np.stack(injections)

    test_data = torch.from_numpy(injections).to(torch.float32)
    test_params = torch.from_numpy(params).to(torch.float32)

    test_dataset = torch.utils.data.TensorDataset(test_data, test_params)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        pin_memory=False if device == "cpu" else True,
        batch_size=1,
        pin_memory_device=device,
    )

    bilby_results_dir = datadir / "bilby" / "rundir" / "final_result"
    bilby_results_paths = sorted(list(bilby_results_dir.iterdir()))

    # get index of phi samples to use later when converting to ra
    phi_index = inference_params.index("phi")
    inference_params[phi_index] = "ra"

    bilby_results, descaled_results, results = [], [], []
    for i, ((signal, param), bilby_result, geocent_time) in enumerate(
        zip(test_dataloader, bilby_results_paths, times)
    ):

        # sample our model on the data
        signal = signal.to(device)
        param = param.to(device)
        strain, scaled_param = preprocessor(signal, param)
        param = param.cpu().numpy()[0]

        # load in the corresponding bilby result
        bilby_result = bilby.result.Result.from_pickle(bilby_result)
        # convert phi to ra
        param[phi_index] = ra_from_phi(param[phi_index], geocent_time)
        # set the ground truth parameters
        bilby_result.injection_parameters = {
            k: float(v) for k, v in zip(inference_params, param)
        }
        bilby_result.label = f"bilby_{i}"
        bilby_results.append(bilby_result)

        _time = time()
        with torch.no_grad():
            samples = flow_obj.flow.sample(num_samples_draw, context=strain)
            descaled_samples = preprocessor.scaler(
                samples[0].transpose(1, 0), reverse=True
            )
            descaled_samples = descaled_samples.unsqueeze(0).transpose(2, 1)
            samples = samples.cpu().numpy()[0]
            descaled_samples = descaled_samples.cpu().numpy()[0]

        _time = time() - _time

        descaled_samples[:, phi_index] = ra_from_phi(
            descaled_samples[:, phi_index], geocent_time
        )

        descaled_res = cast_samples_as_bilby_result(
            descaled_samples, param, inference_params, priors, label="flow"
        )
        descaled_results.append(descaled_res)

        res = cast_samples_as_bilby_result(
            samples,
            scaled_param.cpu().numpy()[0],
            inference_params,
            priors,
            label="flow",
        )
        results.append(res)

    # generate corner plots of the results on top of each other
    results = np.column_stack((descaled_results, bilby_results))
    generate_corner_plots(results, writedir / "corner")
