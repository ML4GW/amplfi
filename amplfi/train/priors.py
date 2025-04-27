from math import pi

import torch
from ml4gw import distributions
from ml4gw.waveforms.conversion import (
    bilby_spins_to_lalsim,
    chirp_mass_and_mass_ratio_to_components,
)
from torch.distributions import Uniform

from .data.utils.utils import ParameterSampler, ParameterTransformer

sg_transformer = ParameterTransformer(hrss=torch.log)

# make the below callables that return parameter samplers
# so that jsonargparse can serialize them properly


# prior and parameter transformer for sg use case
def sg_prior() -> ParameterSampler:
    return ParameterSampler(
        frequency=Uniform(32, 1024),
        quality=Uniform(2, 100),
        hrss=distributions.LogUniform(1e-23, 5e-20),
        phase=Uniform(0, 2 * pi),
        eccentricity=Uniform(0, 1),
    )


def precessing_to_lalsimulation_parameters(
    parameters: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Convert precessing spin parameters to lalsimulation parameters
    """
    mass_1, mass_2 = chirp_mass_and_mass_ratio_to_components(
        parameters["chirp_mass"], parameters["mass_ratio"]
    )

    parameters["mass_1"] = mass_1
    parameters["mass_2"] = mass_2

    # TODO: hard coding f_ref = 40 here b/c not sure best way to link this
    # to the f_ref specified in the config file
    incl, s1x, s1y, s1z, s2x, s2y, s2z = bilby_spins_to_lalsim(
        parameters["inclination"],
        parameters["phi_jl"],
        parameters["tilt_1"],
        parameters["tilt_2"],
        parameters["phi_12"],
        parameters["a_1"],
        parameters["a_2"],
        parameters["mass_1"],
        parameters["mass_2"],
        40,
        torch.zeros(len(mass_1), device=mass_1.device),
    )

    parameters["s1x"] = s1x
    parameters["s1y"] = s1y
    parameters["s1z"] = s1z
    parameters["s2x"] = s2x
    parameters["s2y"] = s2y
    parameters["s2z"] = s2z
    parameters["inclination"] = incl
    return parameters


def aligned_to_lalsimulation_parameters(
    parameters: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Convert precessing spin parameters to lalsimulation parameters
    """
    mass_1, mass_2 = chirp_mass_and_mass_ratio_to_components(
        parameters["chirp_mass"], parameters["mass_ratio"]
    )

    parameters["mass_1"] = mass_1
    parameters["mass_2"] = mass_2

    parameters["s1x"] = torch.zeros_like(mass_1)
    parameters["s1y"] = torch.zeros_like(mass_1)

    parameters["s2x"] = torch.zeros_like(mass_1)
    parameters["s2y"] = torch.zeros_like(mass_1)

    parameters["s1z"] = parameters["chi1"]
    parameters["s2z"] = parameters["chi2"]
    return parameters


# TODO: we want validate_args = False at test time
# but True at train time


# priors and parameter transformers for cbc use case
def aligned_cbc_prior() -> ParameterSampler:
    """
    Prior for aligned-spin CBC waveform generation, e.g with IMRPhenomD
    """
    return ParameterSampler(
        conversion_function=aligned_to_lalsimulation_parameters,
        chirp_mass=Uniform(
            torch.as_tensor(10, dtype=torch.float32),
            torch.as_tensor(100, dtype=torch.float32),
            validate_args=False,
        ),
        mass_ratio=Uniform(
            torch.as_tensor(0.125, dtype=torch.float32),
            torch.as_tensor(0.999, dtype=torch.float32),
            validate_args=False,
        ),
        distance=Uniform(
            torch.as_tensor(100, dtype=torch.float32),
            torch.as_tensor(3100, dtype=torch.float32),
            validate_args=False,
        ),
        inclination=distributions.Sine(
            torch.as_tensor(0, dtype=torch.float32),
            torch.as_tensor(torch.pi, dtype=torch.float32),
            validate_args=False,
        ),
        phic=Uniform(
            torch.as_tensor(0, dtype=torch.float32),
            torch.as_tensor(2 * torch.pi, dtype=torch.float32),
            validate_args=False,
        ),
        chi1=Uniform(
            torch.as_tensor(-0.999, dtype=torch.float32),
            torch.as_tensor(0.999, dtype=torch.float32),
            validate_args=False,
        ),
        chi2=Uniform(
            torch.as_tensor(-0.999, dtype=torch.float32),
            torch.as_tensor(0.999, dtype=torch.float32),
            validate_args=False,
        ),
    )


def precessing_cbc_prior() -> ParameterSampler:
    """
    Prior for precessing-spin CBC waveform generation, e.g with IMRPhenomPv2
    """

    return ParameterSampler(
        conversion_function=precessing_to_lalsimulation_parameters,
        chirp_mass=Uniform(
            torch.as_tensor(10, dtype=torch.float32),
            torch.as_tensor(100, dtype=torch.float32),
            validate_args=False,
        ),
        mass_ratio=Uniform(
            torch.as_tensor(0.125, dtype=torch.float32),
            torch.as_tensor(0.999, dtype=torch.float32),
            validate_args=False,
        ),
        distance=Uniform(
            torch.as_tensor(100, dtype=torch.float32),
            torch.as_tensor(3100, dtype=torch.float32),
            validate_args=False,
        ),
        inclination=distributions.Sine(
            torch.as_tensor(0, dtype=torch.float32),
            torch.as_tensor(torch.pi, dtype=torch.float32),
            validate_args=False,
        ),
        phic=Uniform(
            torch.as_tensor(0, dtype=torch.float32),
            torch.as_tensor(2 * torch.pi, dtype=torch.float32),
            validate_args=False,
        ),
        a_1=Uniform(
            torch.as_tensor(0, dtype=torch.float32),
            torch.as_tensor(0.999, dtype=torch.float32),
            validate_args=False,
        ),
        a_2=Uniform(
            torch.as_tensor(0, dtype=torch.float32),
            torch.as_tensor(0.999, dtype=torch.float32),
            validate_args=False,
        ),
        tilt_1=Uniform(
            torch.as_tensor(0, dtype=torch.float32),
            torch.as_tensor(pi, dtype=torch.float32),
            validate_args=False,
        ),
        tilt_2=Uniform(
            torch.as_tensor(0, dtype=torch.float32),
            torch.as_tensor(pi, dtype=torch.float32),
            validate_args=False,
        ),
        phi_jl=Uniform(
            torch.as_tensor(0, dtype=torch.float32),
            torch.as_tensor(2 * torch.pi, dtype=torch.float32),
            validate_args=False,
        ),
        phi_12=Uniform(
            torch.as_tensor(0, dtype=torch.float32),
            torch.as_tensor(2 * torch.pi, dtype=torch.float32),
            validate_args=False,
        ),
    )


def cbc_testing_delta_function_prior() -> ParameterSampler:
    return ParameterSampler(
        chirp_mass=distributions.DeltaFunction(
            torch.as_tensor(55, dtype=torch.float32),
            validate_args=False,
        ),
        mass_ratio=distributions.DeltaFunction(
            torch.as_tensor(0.9, dtype=torch.float32),
            validate_args=False,
        ),
        distance=distributions.DeltaFunction(
            torch.as_tensor(1000, dtype=torch.float32),
            validate_args=False,
        ),
        inclination=distributions.DeltaFunction(
            torch.as_tensor(torch.pi / 6, dtype=torch.float32),
            validate_args=False,
        ),
        phic=distributions.DeltaFunction(
            torch.as_tensor(torch.pi, dtype=torch.float32),
            validate_args=False,
        ),
        chi1=distributions.DeltaFunction(
            torch.as_tensor(0, dtype=torch.float32),
            validate_args=False,
        ),
        chi2=distributions.DeltaFunction(
            torch.as_tensor(0, dtype=torch.float32),
            validate_args=False,
        ),
    )
