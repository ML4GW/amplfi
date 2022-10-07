from typing import Callable, Dict, List, Optional

import numpy as np
from bilby.gw.waveform_generator import WaveformGenerator


def generate_gw(
    sample_params: Dict[List, str],
    sample_rate: float,
    waveform_duration: float,
    waveform: Callable,
    waveform_arguments: Optional[Dict] = None,
    parameter_conversion: Optional[Callable] = None,
):
    """
    Generate raw frequency domain gravitational-wave signals
    pre-interferometer projection.

    Args:
        sample_params:
            Dictionary of GW parameters where key is the parameter name
            and value is a list of the parameters
        sample_rate:
            Rate at which to sample time series
        waveform_duration:
            Duration of waveform
        waveform:
            Callable whose first argument is a frequency array
            and returns returns frequency domain
            hplus and hcross polarizations at the specified frequencies
            for a given set of parameters that are the following arguments
        waveform_arguments:
            A dictionary of fixed keyword arguments to pass to
            frequency_domain_source_model via bilby waveform generator
        parameter_conversion:
            Callable to convert from sampled parameters
            to parameters of the waveform generator.
            Default value is the identity,
            i.e. it leaves the parameters unaffected.
            Typically used when generating
            CBC waveforms via lal_binary_black_hole

    Returns:
        An (n_samples, 2, waveform_size) array, containing both polarizations
        for each of the desired number of samples.
        The waveforms are shifted such that
        the coalescence time lies at the center of the sample
    """

    sample_params = [
        dict(zip(sample_params, col)) for col in zip(*sample_params.values())
    ]

    n_samples = len(sample_params)

    waveform_generator = WaveformGenerator(
        duration=waveform_duration,
        sampling_frequency=sample_rate,
        frequency_domain_source_model=waveform,
        waveform_arguments=waveform_arguments,
        parameter_conversion=parameter_conversion,
    )

    waveform_size = int(sample_rate * waveform_duration)

    num_pols = 2
    signals = np.zeros((n_samples, num_pols, waveform_size))

    for i, p in enumerate(sample_params):
        polarizations = waveform_generator.time_domain_strain(p)
        polarization_names = sorted(polarizations.keys())
        polarizations = np.stack(
            [polarizations[p] for p in polarization_names]
        )

        # center so that coalescence time is middle sample
        dt = waveform_duration / 2
        polarizations = np.roll(polarizations, int(dt * sample_rate), axis=-1)
        signals[i] = polarizations

    return signals
