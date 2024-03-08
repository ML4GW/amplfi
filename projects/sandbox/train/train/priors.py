from math import pi

import torch
from torch.distributions import Uniform
from train.data.utils import ParameterSampler, ParameterTransformer

from ml4gw import distributions

# prior and parameter transformer for sg use case
sine_gaussian = ParameterSampler(
    frequency=Uniform(32, 1024),
    quality=Uniform(2, 100),
    hrss=distributions.LogUniform(1e-23, 5e-20),
    phase=Uniform(0, 2 * pi),
    eccentricity=Uniform(0, 1),
)

sine_gaussian_transformer = ParameterTransformer(hrss=torch.log)


# priors and parameter transformers for cbc use case
cbc_prior = ParameterSampler(
    chirp_mass=torch.distributions.Uniform(
        torch.as_tensor(10, dtype=torch.float32),
        torch.as_tensor(1000, dtype=torch.float32),
    ),
    mass_ratio=torch.distributions.Uniform(
        torch.as_tensor(0.125, dtype=torch.float32),
        torch.as_tensor(0.999, dtype=torch.float32),
    ),
    distance=distributions.PowerLaw(
        torch.as_tensor(10, dtype=torch.float32),
        torch.as_tensor(30, dtype=torch.float32),
        index=2,
    ),
    inclination=distributions.Sine(
        torch.as_tensor(0, dtype=torch.float32),
        torch.as_tensor(torch.pi, dtype=torch.float32),
    ),
    phic=torch.distributions.Uniform(
        torch.as_tensor(0, dtype=torch.float32),
        torch.as_tensor(2 * torch.pi, dtype=torch.float32),
    ),
    chi1=distributions.DeltaFunction(
        torch.as_tensor(0, dtype=torch.float32),
    ),
    chi2=distributions.DeltaFunction(torch.as_tensor(0, dtype=torch.float32)),
)
