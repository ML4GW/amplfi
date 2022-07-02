from typing import Dict

import bilby
import numpy as np
from mlpe.injection.utils import calc_snr


def generate_gw(
    sample_params: Dict[str, np.ndarray],
    domain: str = "time",
    waveform_generator=bilby.gw.waveform_generator.WaveformGenerator,
):
    """Generate raw gravitational-wave signals, pre-interferometer projection.

    Args:
        sample_params: dictionary of GW parameters
        domain: domain in which to generate data ("time" or "frequency")
        waveform_generator: a fully initialized bilby.gw.WaveformGenerator

    Returns:
        An (n_samples, 2, waveform_size) array, containing both polarizations
        for each of the desired number of samples. The first polarization is
        always plus and the second is always cross
    """

    # reformat sample parameters
    sample_params = [
        dict(zip(sample_params, col)) for col in zip(*sample_params.values())
    ]
    n_samples = len(sample_params)

    # extract signal properties from waveform generator
    sample_rate = waveform_generator.sampling_frequency
    waveform_duration = waveform_generator.duration

    if domain == "time":
        waveform_size = int(sample_rate * waveform_duration)

    elif domain == "frequency":
        fmax = sample_rate / 2
        df = 1 / waveform_duration
        waveform_size = int(fmax / df) + 1

    num_pols = 2
    signals = np.zeros((n_samples, num_pols, waveform_size))

    for i, p in enumerate(sample_params):

        # generate data based on domain requested
        if domain == "time":
            polarizations = waveform_generator.time_domain_strain(p)
        elif domain == "frequency":
            polarizations = waveform_generator.frequency_domain_strain(p)

            polarization_names = sorted(polarizations.keys())

            polarizations = np.stack(
                [polarizations[p] for p in polarization_names]
            )

            signals[i] = polarizations

    return signals


def project_raw_gw(
    raw_waveforms: np.ndarray,
    sample_params: Dict[str, np.ndarray],
    sample_rate: float,
    ifo: str,
    get_snr: bool = False,
    noise_psd=None,
):
    """Project a raw gravitational wave onto an intterferometer

    Args:
        raw_waveforms: the plus and cross polarizations of a list of GWs
        sample_params: dictionary of GW parameters
        waveform_generator: the waveform generator that made the raw GWs
        ifo: interferometer
        get_snr: return the SNR of each sample
        noise_psd: background noise PSD used to calculate SNR the sample

    Returns:
        An (n_samples, waveform_size) array containing the GW signals as they
        would appear in the given interferometer with the given set of sample
        parameters. If get_snr=True, also returns a list of the SNR associated
        with each signal
    """

    polarizations = {
        "plus": raw_waveforms[:, 0, :],
        "cross": raw_waveforms[:, 1, :],
    }

    sample_params = [
        dict(zip(sample_params, col)) for col in zip(*sample_params.values())
    ]
    n_sample = len(sample_params)

    waveform_duration = raw_waveforms.shape[-1] / sample_rate
    waveform_size = int(sample_rate * waveform_duration)

    signals = np.zeros((n_sample, waveform_size))
    snr = np.zeros(n_sample)

    ifo = bilby.gw.detector.get_empty_interferometer(ifo)
    for i, p in enumerate(sample_params):

        # For less ugly function calls later on
        ra = p["ra"]
        dec = p["dec"]
        geocent_time = p["geocent_time"]
        psi = p["psi"]

        # Generate signal in IFO
        signal = np.zeros(waveform_size)
        for mode, polarization in polarizations.items():
            # Get ifo response
            response = ifo.antenna_response(ra, dec, geocent_time, psi, mode)
            signal += response * polarization[i]

        # Total shift = shift to trigger time + geometric shift
        dt = waveform_duration / 2.0
        dt += ifo.time_delay_from_geocenter(ra, dec, geocent_time)
        signal = np.roll(signal, int(np.round(dt * sample_rate)))

        # Calculate SNR
        if noise_psd is not None:
            if get_snr:
                snr[i] = calc_snr(signal, noise_psd, sample_rate)

        signals[i] = signal

    if get_snr:
        return signals, snr
    return signals
