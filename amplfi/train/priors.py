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
    chirp_mass, mass_ratio = (
        parameters.pop("chirp_mass"),
        parameters.pop("mass_ratio"),
    )
    mass_1, mass_2 = chirp_mass_and_mass_ratio_to_components(
        chirp_mass, mass_ratio
    )

    a_1, a_2 = parameters.pop("a_1"), parameters.pop("a_2")
    tilt_1, tilt_2 = parameters.pop("tilt_1"), parameters.pop("tilt_2")
    inclination = parameters.pop("inclination")
    phi_jl, phi_12 = parameters.pop("phi_jl"), parameters.pop("phi_12")
    phic = parameters.pop("phic")

    # TODO: hard coding f_ref = 40 here b/c not sure best way to link this
    # to the f_ref specified in the config file
    incl, s1x, s1y, s1z, s2x, s2y, s2z = bilby_spins_to_lalsim(
        inclination,
        phi_jl,
        tilt_1,
        tilt_2,
        phi_12,
        a_1,
        a_2,
        mass_1,
        mass_2,
        40,
        torch.zeros(len(inclination), device=inclination.device),
    )

    output = {}
    output["mass_1"] = mass_1
    output["mass_2"] = mass_2
    output["chirp_mass"] = chirp_mass
    output["mass_ratio"] = mass_ratio
    output["s1x"] = s1x
    output["s1y"] = s1y
    output["s1z"] = s1z
    output["s2x"] = s2x
    output["s2y"] = s2y
    output["s2z"] = s2z
    output["distance"] = parameters["distance"]
    output["inclination"] = incl
    output["phic"] = phic
    return output


def aligned_to_lalsimulation_parameters(
    parameters: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Convert precessing spin parameters to lalsimulation parameters
    """
    chirp_mass, mass_ratio = (
        parameters.pop("chirp_mass"),
        parameters.pop("mass_ratio"),
    )
    mass_1, mass_2 = chirp_mass_and_mass_ratio_to_components(
        chirp_mass, mass_ratio
    )

    s1x = torch.zeros_like(mass_1)
    s1y = torch.zeros_like(mass_1)

    s2x = torch.zeros_like(mass_1)
    s2y = torch.zeros_like(mass_1)

    output = {}
    output["chi1"] = parameters["chi1"]
    output["chi2"] = parameters["chi2"]
    output["mass_1"] = mass_1
    output["mass_2"] = mass_2
    output["chirp_mass"] = chirp_mass
    output["mass_ratio"] = mass_ratio
    output["s1x"] = s1x
    output["s1y"] = s1y
    output["s1z"] = parameters.pop("chi1")
    output["s2x"] = s2x
    output["s2y"] = s2y
    output["s2z"] = parameters.pop("chi2")
    output["distance"] = parameters.pop("distance")
    output["inclination"] = parameters.pop("inclination")
    output["phic"] = parameters.pop("phic")
    return output


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
