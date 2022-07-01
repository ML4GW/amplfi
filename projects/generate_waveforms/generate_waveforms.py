from hermes.typeo import typeo
from typing import Callable, Optional
from pathlib import Path
import bilby
import bbhnet.injection as injection

@typeo
def main(
    waveform: Callable,
    prior: str,
    sample_rate: float,
    domain: str
    n_samples: int ,
    waveform_duration: float,
    datadir: Path,
    logdir: Path,
    inference_params: Optional[List[str]] = None,
    **waveform_arguments 
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
            If not passed will default to all the parameters in the prior
        **waveform_arguments:
            Additional arguments to pass to waveform generator, that will ultimately get passed
            to the waveform callable specified. For example,
            generating BBH waveforms requires the specification of a 
            waveform_approximant
    """
    
    # make data dir
    datadir.mkdir(exist_ok=True, parents=True)  
    
    # define a bilby waveform generator
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=waveform_duration,
        sampling_frequency=sample_rate,
        frequency_domain_source_model=waveform,
        parameter_conversion=convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments
    )     

    # initiate prior and sample
    priors = bilby.gw.prior.PriorDict(prior_file)
    sample_params = priors.sample(n_samples)

    signals = injection.generate_gw(sample_params, waveform_generator=waveform_generator) 
     
