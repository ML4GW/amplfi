import numpy as np
from bilby.core.prior import Constraint, Cosine, LogUniform, PriorDict, Sine, Uniform
from bilby.gw.prior import UniformSourceFrame

from ml4gw import distributions
from ml4gw.waveforms.generator import ParameterSampler
import torch


def sg_uniform():
    prior_dict = PriorDict()
    prior_dict["quality"] = Uniform(
        name="quality", minimum=2, maximum=100, latex_label="quality"
    )
    prior_dict["frequency"] = Uniform(
        name="frequency", minimum=32, maximum=1024, latex_label="frequency"
    )
    prior_dict["dec"] = Cosine(name="dec", latex_label="dec")

    prior_dict["ra"] = Uniform(
        name="ra", minimum=0, maximum=2 * np.pi, latex_label="ra"
    )
    prior_dict["hrss"] = LogUniform(
        name="hrss", minimum=1e-23, maximum=5e-20, latex_label="hrss"
    )
    prior_dict["time"] = 0

    prior_dict["eccentricity"] = Uniform(
        name="eccentricity", minimum=0, maximum=1, latex_label="eccentricity"
    )
    prior_dict["psi"] = Uniform(
        name="psi", minimum=0, maximum=np.pi, latex_label="psi"
    )
    prior_dict["phase"] = Uniform(
        name="phase", minimum=0, maximum=2 * np.pi, latex_label="phase"
    )

    return prior_dict


def nonspin_bbh_component_mass_parameter_sampler(device='cpu'):
    return ParameterSampler(
        mass_1=distributions.Uniform(
            torch.as_tensor(30, device=device, dtype=torch.float32),
            torch.as_tensor(40, device=device, dtype=torch.float32),
            name="mass_1"
        ),
        mass_2=distributions.Uniform(
            torch.as_tensor(20, device=device, dtype=torch.float32),
            torch.as_tensor(30, device=device, dtype=torch.float32),
            name="mass_2"
        ),
        luminosity_distance=distributions.PowerLaw(
            torch.as_tensor(10, device=device, dtype=torch.float32),
            torch.as_tensor(1000, device=device, dtype=torch.float32),
            index=2,
            name="luminosity_distance"
        ),
        dec=distributions.Cosine(
            torch.as_tensor(-torch.pi/2, device=device, dtype=torch.float32),
            torch.as_tensor(torch.pi/2, device=device, dtype=torch.float32),
            name="dec"
        ),
        phi=distributions.Uniform(
            torch.as_tensor(0, device=device, dtype=torch.float32),
            torch.as_tensor(2*torch.pi, device=device, dtype=torch.float32),
            name="phi"
        ),
        theta_jn=distributions.Sine(
            torch.as_tensor(0, device=device, dtype=torch.float32),
            torch.as_tensor(torch.pi, device=device, dtype=torch.float32),
            name="theta_jn"
        ),
        psi=distributions.Uniform(
            torch.as_tensor(0, device=device, dtype=torch.float32),
            torch.as_tensor(torch.pi, device=device, dtype=torch.float32),
            name="psi"
        ),
        phase=distributions.Uniform(
            torch.as_tensor(0, device=device, dtype=torch.float32),
            torch.as_tensor(2 * torch.pi, device=device, dtype=torch.float32),
            name="phase"
        ),
        a_1=distributions.DeltaFunction(
            torch.as_tensor(0, device=device, dtype=torch.float32), name="a_1"
        ),
        a_2=distributions.DeltaFunction(
            torch.as_tensor(0, device=device, dtype=torch.float32), name="a_2"
        ),
    )


def nonspin_bbh_chirp_mass_q_parameter_sampler(device='cpu'):
    return ParameterSampler(
        chirp_mass=distributions.Uniform(
            torch.as_tensor(20, device=device, dtype=torch.float32),
            torch.as_tensor(40, device=device, dtype=torch.float32),
            name="chirp_mass"
        ),
        mass_ratio=distributions.Uniform(
            torch.as_tensor(0.125, device=device, dtype=torch.float32),
            torch.as_tensor(0.999, device=device, dtype=torch.float32),
            name="mass_ratio"
        ),
        luminosity_distance=distributions.PowerLaw(
            torch.as_tensor(10, device=device, dtype=torch.float32),
            torch.as_tensor(1000, device=device, dtype=torch.float32),
            index=2,
            name="luminosity_distance"
        ),
        dec=distributions.Cosine(
            torch.as_tensor(-torch.pi/2, device=device, dtype=torch.float32),
            torch.as_tensor(torch.pi/2, device=device, dtype=torch.float32),
            name="dec"
        ),
        phi=distributions.Uniform(
            torch.as_tensor(0, device=device, dtype=torch.float32),
            torch.as_tensor(2*torch.pi, device=device, dtype=torch.float32),
            name="phi"
        ),
        theta_jn=distributions.Sine(
            torch.as_tensor(0, device=device, dtype=torch.float32),
            torch.as_tensor(torch.pi, device=device, dtype=torch.float32),
            name="theta_jn"
        ),
        psi=distributions.Uniform(
            torch.as_tensor(0, device=device, dtype=torch.float32),
            torch.as_tensor(torch.pi, device=device, dtype=torch.float32),
            name="psi"
        ),
        phase=distributions.Uniform(
            torch.as_tensor(0, device=device, dtype=torch.float32),
            torch.as_tensor(2 * torch.pi, device=device, dtype=torch.float32),
            name="phase"
        ),
        a_1=distributions.DeltaFunction(
            torch.as_tensor(0, device=device, dtype=torch.float32), name="a_1"
        ),
        a_2=distributions.DeltaFunction(
            torch.as_tensor(0, device=device, dtype=torch.float32), name="a_2"
        ),
    )


def nonspin_bbh_chirp_mass_q():
    prior_dict = PriorDict()
    prior_dict["mass_1"] = Constraint(name="mass_1", minimum=10, maximum=80)
    prior_dict["mass_2"] = Constraint(name="mass_2", minimum=10, maximum=80)
    prior_dict["mass_ratio"] = Uniform(
        name="mass_ratio", minimum=0.125, maximum=0.999
    )
    prior_dict["chirp_mass"] = Uniform(
        name="chirp_mass", minimum=20, maximum=35
    )
    prior_dict["luminosity_distance"] = UniformSourceFrame(
        name="luminosity_distance", minimum=10, maximum=3000, unit="Mpc"
    )
    prior_dict["dec"] = Cosine(name="dec")
    prior_dict["phi"] = Uniform(
        name="phi", minimum=0, maximum=2 * np.pi, boundary="periodic"
    )
    prior_dict["theta_jn"] = Sine(name='theta_jn')
    prior_dict["psi"] = Uniform(
        name="psi", minimum=0, maximum=np.pi, latex_label="psi"
    )
    prior_dict["phase"] = Uniform(
        name="phase", minimum=0, maximum=2 * np.pi, latex_label="phase"
    )
    prior_dict["a_1"] = 0
    prior_dict["a_2"] = 0
    prior_dict["tilt_1"] = 0
    prior_dict["tilt_2"] = 0
    prior_dict["phi_12"] = 0
    prior_dict["phi_jl"] = 0

    return prior_dict

def nonspin_bbh_component_mass():
    prior_dict = PriorDict()
    prior_dict["mass_1"] = Uniform(20, 40, name="mass_1")
    prior_dict["mass_2"] = Uniform(20, 40, name="mass_2")
    prior_dict["mass_ratio"] = Constraint(
        name="mass_ratio", minimum=0.2, maximum=0.999
    )
    prior_dict["luminosity_distance"] = UniformSourceFrame(
        name="luminosity_distance", minimum=10, maximum=100, unit="Mpc"
    )
    prior_dict["dec"] = Cosine(name="dec")
    prior_dict["phi"] = Uniform(
        name="phi", minimum=0, maximum=2 * np.pi, boundary="periodic"
    )
    prior_dict["theta_jn"] = Sine(name='theta_jn')
    prior_dict["psi"] = Uniform(
        name="psi", minimum=0, maximum=np.pi, latex_label="psi"
    )
    prior_dict["phase"] = Uniform(
        name="phase", minimum=0, maximum=2 * np.pi, latex_label="phase"
    )
    prior_dict["a_1"] = 0
    prior_dict["a_2"] = 0
    prior_dict["tilt_1"] = 0
    prior_dict["tilt_2"] = 0
    prior_dict["phi_12"] = 0
    prior_dict["phi_jl"] = 0

    return prior_dict
