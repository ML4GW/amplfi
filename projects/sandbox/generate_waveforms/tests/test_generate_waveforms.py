from pathlib import Path

import bilby
import h5py
import mlpe.injection.waveforms as waveforms
import pytest
from bilby.gw.source import lal_binary_black_hole
from generate_waveforms import main


@pytest.fixture(params=["time", "frequency"])
def domain(request):
    return request.param


@pytest.fixture(params=["priors/nonspin_BBH.prior"])
def cbc_prior_file(request):
    return str(Path(__file__).resolve().parent / request.param)


@pytest.fixture(params=["priors/sine_gaussian.prior"])
def sg_prior_file(request):
    return str(Path(__file__).resolve().parent / request.param)


@pytest.fixture(params=[1, 2, 4])
def waveform_duration(request):
    return request.param


@pytest.fixture(params=[2048, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[100])
def n_samples(request):
    return request.param


@pytest.fixture(params=[None, ["ra", "dec"]])
def inference_parameters(request):
    return request.param


def test_generate_waveforms_custom_model(
    domain,
    sample_rate,
    n_samples,
    waveform_duration,
    datadir,
    logdir,
    sg_prior_file,
    inference_parameters,
):

    # extract all the parameters used to generate model
    priors = bilby.gw.prior.PriorDict(sg_prior_file)

    # if inference parameters are not passed,
    # they should be all parameters in prior
    # that are not constraints

    constraints = priors.constraint_keys
    all_params = [param for param in priors.keys() if param not in constraints]
    if inference_parameters is None:
        inference_parameters = all_params

    signal_file = main(
        sg_prior_file,
        waveforms.sine_gaussian_frequency,
        sample_rate,
        domain,
        n_samples,
        waveform_duration,
        datadir,
        logdir,
        inference_params=inference_parameters,
        waveform_arguments=None,
    )

    # dynamically infer shape based on
    # domain
    if domain == "time":
        waveform_size = sample_rate * waveform_duration

    elif domain == "frequency":
        df = 1 / waveform_duration
        fmax = sample_rate / 2
        waveform_size = int(fmax / df) + 1

    with h5py.File(signal_file) as f:
        # parameters to use for inference
        inference_params = f["inference_parameters"]

        # parameters used to generate waveform,
        # but not included in inference
        generation_params = f["parameters"]

        # ensure proper lengths
        assert len(inference_params) == len(inference_parameters)
        assert len(generation_params) == len(all_params) - len(
            inference_params
        )

        # make sure parameters we requested for inference
        # are stored in inference_parameters group
        for param in inference_parameters:

            assert param in inference_params.keys()
            assert inference_params[param].shape == (n_samples,)

        # make sure other parameters used for
        # generation are saved in parameters group
        for param in all_params:
            if param not in inference_parameters:
                assert param in generation_params.keys()
                assert generation_params[param].shape == (n_samples,)

        signals = f["signals"][()]
        expected_shape = (n_samples, 2, waveform_size)
        assert signals.shape == expected_shape


def test_generate_waveforms_cbc_model(
    domain,
    sample_rate,
    n_samples,
    waveform_duration,
    datadir,
    logdir,
    cbc_prior_file,
    inference_parameters,
):

    # extract all the parameters used to generate model
    priors = bilby.gw.prior.PriorDict(cbc_prior_file)

    constraints = priors.constraint_keys
    all_params = [param for param in priors.keys() if param not in constraints]
    if inference_parameters is None:
        inference_parameters = all_params

    signal_file = main(
        cbc_prior_file,
        lal_binary_black_hole,
        sample_rate,
        domain,
        n_samples,
        waveform_duration,
        datadir,
        logdir,
        inference_params=inference_parameters,
        waveform_arguments={
            "waveform_approximant": "IMRPhenomPv2",
            "reference_frequency": 50,
            "minimum_frequency": 20,
        },
    )

    # dynamically infer shape based on
    # domain
    if domain == "time":
        waveform_size = sample_rate * waveform_duration

    elif domain == "frequency":
        df = 1 / waveform_duration
        fmax = sample_rate / 2
        waveform_size = int(fmax / df) + 1

    with h5py.File(signal_file) as f:
        # parameters to use for inference
        inference_params = f["inference_parameters"]

        print(inference_params.keys())
        # parameters used to generate waveform,
        # but not included in inference
        generation_params = f["parameters"]

        # ensure proper lengths
        assert len(inference_params) == len(inference_parameters)
        assert len(generation_params) == len(all_params) - len(
            inference_params
        )

        # make sure parameters we requested for inference
        # are stored in inference_parameters group
        for param in inference_parameters:

            assert param in inference_params.keys()
            assert inference_params[param].shape == (n_samples,)

        # make sure other parameters used for
        # generation are saved in parameters group
        for param in all_params:
            if param not in inference_parameters:
                assert param in generation_params.keys()
                assert generation_params[param].shape == (n_samples,)

        signals = f["signals"][()]
        expected_shape = (n_samples, 2, waveform_size)
        assert signals.shape == expected_shape
