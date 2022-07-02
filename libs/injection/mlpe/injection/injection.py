from typing import Dict

import bilby
import numpy as np


def generate_gw(
    sample_params: Dict[str, np.ndarray],
    domain: str = "time",
    waveform_generator=bilby.gw.waveform_generator.WaveformGenerator,
):
    """Generate raw gravitational-wave signals, pre-interferometer projection.

    Args:
        sample_params: dictionary of GW parameters produced by
            the bilby.PriorDict sample function
        domain: domain in which to generate data ("time" or "frequency")
        waveform_generator:
            a fully initialized bilby.gw.waveform_generator.WaveformGenerator

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

    # infer waveform size depending on
    # domain
    if domain == "time":
        waveform_size = int(sample_rate * waveform_duration)

        signals = np.zeros((n_samples, 2, waveform_size))

    elif domain == "frequency":
        fmax = sample_rate / 2
        df = 1 / waveform_duration
        waveform_size = int(fmax / df) + 1
        signals = np.zeros((n_samples, 2, waveform_size), dtype=np.complex)

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
    duration: float,
    ifo: str,
    domain: str,
):
    """Project a raw gravitational wave onto an interferometer

    Args:
        raw_waveforms: an array of size (n_samples, 2, waveform_size)
        sample_params: parameters produced by bilby prior sample function.
            must include ra, dec geocen_time and psi
        duration: length of the waveform
        ifo: which ifo projecting data onto to
        domain: domain of raw_waveform data (time or frequency)

    Returns:
        An (n_samples, waveform_size) array containing the GW signals as they
        would appear in the given interferometer with the given set of sample
        parameters. If get_snr=True, also returns a list of the SNR associated
        with each signal
    """

    # format polarizations
    polarizations = {
        "plus": raw_waveforms[:, 0, :],
        "cross": raw_waveforms[:, 1, :],
    }

    # format sample params
    sample_params = [
        dict(zip(sample_params, col)) for col in zip(*sample_params.values())
    ]
    n_sample = len(sample_params)

    # infer waveform parameters from passed array
    waveform_size = raw_waveforms.shape[-1]

    # initiate signal array
    signals = np.zeros((n_sample, waveform_size))

    ifo = bilby.gw.detector.get_empty_interferometer(ifo)
    for i, p in enumerate(sample_params):

        # unpack sky loc params
        ra = p["ra"]
        dec = p["dec"]
        geocent_time = p["geocent_time"]
        psi = p["psi"]

        # calculate dt shift for this ifo;
        # first shift signal to center,
        # then apply shift from geocenter
        dt = duration / 2.0
        dt += ifo.time_delay_from_geocenter(ra, dec, geocent_time)

        # generate signal in ifo
        signal = np.zeros(waveform_size)
        for mode, polarization in polarizations.items():

            # get ifo response
            response = ifo.antenna_response(ra, dec, geocent_time, psi, mode)
            signal += response * polarization[i]

        # apply shift
        if domain == "time":
            sample_rate = waveform_size / duration
            signal = np.roll(signal, int(np.round(dt * sample_rate)))

        elif domain == "frequency":
            df = 1 / duration
            fmax = waveform_size * df
            frequencies = np.linspace(0, fmax, df)
            # shift in frequency domain is phase shift
            signal = signal * np.exp(-1j * 2 * np.pi * dt * frequencies)

        signals[i] = signal

    return signals
