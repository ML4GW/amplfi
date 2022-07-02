from pathlib import Path
from typing import Callable, List, Optional

import bilby
import h5py
import mlpe.injection as injection
import numpy as np
from hermes.typeo import typeo


@typeo
def main(
    waveform: Callable,
    prior_file: str,
    sample_rate: float,
    domain: str,
    n_samples: int,
    waveform_duration: float,
    datadir: Path,
    logdir: Path,
    inference_params: Optional[List[str]] = None,
    **waveform_arguments,
):
    """Generates a dataset of raw waveforms. The goal was to make this
    project waveform agnositic

    Args:
        waveform: A callable compatible with bilby waveform generator
        prior: path to prior for generating waveforms
        sample_rate: sample rate for generating waveform
        domain: what domain to create data in (time or frequency)
        n_samples: number of signal to inject
        waveform_duration: length of injected waveforms
        datadir: Path to store data
        logdir: Path to store logs
        inference_params:
            List of parameters on which to perform inference.
            If not passed will default to all the parameters in the prior.
            The idea is that there may be use cases where we want
            to fix some of the parameters,
            or not use some of the parameters in inference.

        **waveform_arguments:
            Additional arguments to pass to waveform generator,
            that will ultimately get passed
            to the waveform callable specified. For example,
            generating BBH waveforms requires the specification of a
            waveform_approximant
    """

    # make data dir
    datadir.mkdir(exist_ok=True, parents=True)
    signal_file = datadir / "signals"

    # define a bilby waveform generator
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=waveform_duration,
        sampling_frequency=sample_rate,
        frequency_domain_source_model=waveform,
        waveform_arguments=waveform_arguments,
    )

    # initiate prior and sample
    priors = bilby.gw.prior.PriorDict(prior_file)
    sample_params = priors.sample(n_samples)

    # if inference params are not passed
    # initalize them to all params in prior
    inference_params = inference_params or list(priors.keys())

    # validate the inference params
    for param in inference_params:
        if param not in priors.keys():
            raise ValueError(
                f"Inference param {param} is not found in the prior"
            )

    # generate signals
    # TODO: implement the ability to
    # specify domain of data in injection library
    signals = injection.generate_gw(sample_params, domain, waveform_generator)

    # sanity check for nan values
    if np.isnan(signals).any():
        raise ValueError("The signals contain NaN values")

    # write signals, params used to generate them,
    # as well as the inference params to a separate
    # group
    with h5py.File(signal_file, "w") as f:

        # create groups for parameters used to
        # generate data, but not used for inference,
        # and parameters used for inference.

        param_group = f.create_group("parameters")
        inference_param_group = f.create_group("inference_parameters")
        for k, v in sample_params.items():
            if k in inference_params:
                inference_param_group.create_dataset(k, data=v)
            else:
                param_group.create_dataset(k, data=v)

        f.create_dataset("signals", data=signals)

        # write attributes
        f.attrs.update(
            {
                "size": n_samples,
                "sample_rate": sample_rate,
                "waveform_duration": waveform_duration,
            }
        )
