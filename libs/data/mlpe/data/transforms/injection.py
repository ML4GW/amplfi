from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ml4gw import gw

from typing import Callable, List

import torch

from ml4gw import gw
from ml4gw.waveforms.generator import ParameterSampler


class PEInjector(torch.nn.Module):
    """
    A torch module that generates waveforms and injects them into
    background kernels
    """

    def __init__(
        self,
        sample_rate: float,
        ifos: List[str],
        intrinsic_parameter_sampler: "ParameterSampler",
        dec: Callable,
        psi: Callable,
        phi: Callable,
        waveform: Callable,
    ):
        super().__init__()

        self.waveform = waveform
        self.parameter_sampler = intrinsic_parameter_sampler
        self.sample_rate = sample_rate

        self.dec = dec
        self.psi = psi
        self.phi = phi

        tensors, vertices = gw.get_ifo_geometry(*ifos)
        self.register_buffer("tensors", tensors)
        self.register_buffer("vertices", vertices)

    def transform(self, parameters: dict):
        # take logarithm of hrss since that is what we train with;
        # remove eccentricity from parameters since we don't train with it
        parameters["hrss"] = torch.log(parameters["hrss"])
        parameters.pop("eccentricity")        
        return parameters

    def sample_waveforms(self, N: int):
        # randomly sample intrinsic parameters and generate raw polarizations
        parameters = self.parameter_sampler(N, device=self.tensors.device)
        waveforms = self.waveform(**parameters)

        n_waveforms = len(waveforms)
        dec, psi, phi = (
            self.dec(n_waveforms),
            self.psi(n_waveforms),
            self.phi(n_waveforms),
        )

        dec = dec.to(waveforms.device)
        psi = psi.to(waveforms.device)
        phi = phi.to(waveforms.device)

        plus, cross = waveforms.transpose(1, 0)
        waveforms = gw.compute_observed_strain(
            dec,
            psi,
            phi,
            detector_tensors=self.tensors,
            detector_vertices=self.vertices,
            sample_rate=self.sample_rate,
            plus=plus,
            cross=cross,
        )
        
        transformed = self.transform(parameters)
        # concatenate intrinsic parameters with sampled extrinsic parameters
        transformed = torch.column_stack(list(transformed.values()))
        transformed = torch.column_stack((transformed, dec, psi, phi))

        return waveforms, transformed

    def forward(self, X: "gw.WaveformTensor"):
        if self.training:
            # inject waveforms into every kernel
            N = len(X)

            # infer kernel size from background
            kernel_size = X.shape[-1]
            waveforms, parameters = self.sample_waveforms(N)

            # calculate the fixed location
            # where waveform T_c will placed
            center = waveforms.shape[-1] // 2
            start = center - (kernel_size // 2)
            stop = center + (kernel_size // 2)

            waveforms = waveforms[:, :, start:stop]
            X += waveforms

            return X, parameters
        return X
