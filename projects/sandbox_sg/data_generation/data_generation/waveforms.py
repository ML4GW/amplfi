import logging
from pathlib import Path
from typing import Callable

import h5py
import numpy as np

from mlpe.injection import generate_time_domain_sine_gaussian
from mlpe.logging import configure_logging
from typeo import scriptify


@scriptify
def main(
    prior: Callable,
    sample_rate: float,
    n_samples: int,
    waveform_duration: float,
    datadir: Path,
    logdir: Path,
    force_generation: bool = False,
    verbose: bool = False,
):
    """Generates a dataset of raw sine gaussian waveforms to use for validation

    Args:

        prior: Callable that instantiates a bilby prior
        waveform: A callable compatible with bilby waveform generator
        sample_rate: sample rate for generating waveform
        n_samples: number of signal to inject
        waveform_duration: length of injected waveforms
        datadir: Path to store data
        logdir: Path to store logs
        waveform_arguments:
            Additional arguments to pass to waveform generator,
            that will ultimately get passed
            to the waveform callable specified. For example,
            generating BBH waveforms requires the specification of a
            waveform_approximant
    """

    configure_logging(logdir / "generate_waveforms.log", verbose)
    datadir.mkdir(exist_ok=True, parents=True)
    signal_file = datadir / "validation_signals.h5"

    if signal_file.exists() and not force_generation:
        logging.info(
            "Signal data already exists and forced generation is off. "
            "Not generating signals."
        )
        return signal_file

    priors = prior()

    params = priors.sample(n_samples)

    cross, plus = generate_time_domain_sine_gaussian(
        frequencies=params["frequency"],
        hrss=params["hrss"],
        qualities=params["quality"],
        phases=params["phase"],
        eccentricities=params["eccentricity"],
        sample_rate=sample_rate,
        duration=waveform_duration,
    )

    signals = np.stack([cross, plus], axis=1)
    # write signals and parameters used to generate them
    with h5py.File(signal_file, "w") as f:

        f.create_dataset("signals", data=signals)

        for k, v in params.items():
            f.create_dataset(k, data=v)

        # write attributes
        f.attrs.update(
            {
                "size": n_samples,
                "sample_rate": sample_rate,
                "waveform_duration": waveform_duration,
            }
        )

    return signal_file


if __name__ == "__main__":
    main()
