from typing import TYPE_CHECKING, Callable, Dict

import torch
from train.data.base import BaseDataset
from train.data.utils import ParameterSampler

from ml4gw.waveforms import SineGaussian

if TYPE_CHECKING:
    from ml4gw.transforms import ChannelWiseScaler

from train.waveforms import WaveformGenerator


class WaveformGeneratorDataset(BaseDataset):
    """
    Dataset that generates waveforms on the fly with torch
    and injects into background noise

    Args:
        waveform_generator:
            Callable that takes a dictionary of Tensors, each of length `N`,
            as input and returns a Tensor containing `N`
            waveforms of shape `(N, num_ifos, strain_dim)`
        parameter_sampler:
            Import path to a Callable that takes an integer `N` as input and
            returnes a dictionary of Tensors, each of length `N`
        num_val_waveforms:
            Total number of validaton waveforms to use.
            This total will be split up among all devices
        num_test_waveforms:
            Total number of test waveforms to use.
            Testing is currently done on a single device
        num_fit_params: N
            Number of parameters to use for fitting the standard scaler
    """

    def __init__(
        self,
        *args,
        parameter_sampler: ParameterSampler,
        num_val_waveforms: int = 10000,
        num_test_waveforms: int = 1000,
        num_fit_params: int = 100000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_val_waveforms = num_val_waveforms
        self.num_fit_params = num_fit_params
        self.num_test_waveforms = num_test_waveforms
        self.parameter_sampler = parameter_sampler
        self.waveform_generator = self.get_waveform_generator()

    def get_waveform_generator(
        self,
    ) -> Callable[[Dict[str, torch.Tensor]], torch.Tensor]:
        raise NotImplementedError

    @property
    def val_waveforms_per_device(self):
        world_size, _ = self.get_world_size_and_rank()
        return self.num_val_waveforms // world_size

    def fit_scaler(self, scaler: "ChannelWiseScaler") -> "ChannelWiseScaler":
        # sample parameters from parameter sampler
        # so we can fit the standard scaler
        parameters = self.parameter_sampler(self.num_fit_params)
        dec, phi, psi = self.sample_extrinsic(
            self.num_fit_params, device="cpu"
        )
        parameters.update({"dec": dec, "phi": phi, "psi": psi})

        # downselect only to those requested to do inference on
        fit_params = {
            k: v for k, v in parameters.items() if k in self.inference_params
        }

        # transform any relevant parameters and fit the scaler
        transformed = self.transform(fit_params)
        transformed = [v for k, v in transformed.items()]
        parameters = torch.vstack(transformed)

        scaler.fit(parameters)
        return scaler

    def get_waveforms(self, num_waveforms: int):
        cross, plus, parameters = self.sample_waveforms(
            num_waveforms, device="cpu"
        )
        self.waveform_generator = self.waveform_generator.to("cpu")
        dec, phi, psi = self.sample_extrinsic(num_waveforms, device="cpu")
        waveforms = self.projector(dec, phi, psi, cross=cross, plus=plus)
        parameters.update({"dec": dec, "phi": phi, "psi": psi})
        parameters = self.transform(parameters)
        return waveforms, parameters

    def get_val_waveforms(self):
        return self.get_waveforms(self.val_waveforms_per_device)

    def get_test_waveforms(self):
        return self.get_waveforms(self.num_test_waveforms)

    def sample_waveforms(self, N: int, device: torch.device):
        # sample intrinsic parameters and generate
        # intrinsic h+ and hx polarizations
        parameters = self.parameter_sampler(N, device=device)
        cross, plus = self.waveform_generator(**parameters)
        cross, plus = cross.float(), plus.float()
        return cross, plus, parameters


class SgGeneratorDataset(WaveformGeneratorDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_waveform_generator(self):
        return SineGaussian(self.sample_rate, self.sample_length)

    def slice_waveforms(self, waveforms: torch.Tensor):
        # for sine gaussians, place waveform in center of kernel
        center = waveforms.shape[-1] // 2
        start = center - (self.pre_whiten_size // 2)
        stop = center + (self.pre_whiten_size // 2)
        return waveforms[..., start:stop]


class CBCGeneratorDataset(WaveformGeneratorDataset):
    def __init__(
        self,
        *args,
        approximant: Callable,
        f_min: float = 0.0,
        f_max: float = 0.0,
        f_ref: float = 40.0,
        padding: float = 0.0,
        **kwargs,
    ):

        self.f_ref = f_ref
        self.approximant = approximant
        self.f_min = f_min
        self.f_max = f_max
        self.padding = padding
        super().__init__(*args, **kwargs)

    def get_waveform_generator(self) -> WaveformGenerator:
        return WaveformGenerator(
            self.approximant,
            self.sample_length,
            self.sample_rate,
            self.f_min,
            self.f_max,
            self.padding,
            waveform_arguments={"f_ref": self.f_ref},
        )

    def slice_waveforms(self, waveforms: torch.Tensor):
        # for cbc waveforms, the padding (see above)
        # determines where the coalescence time lies
        # relative to the right edge, so just subtract
        # the pre-whiten kernel size from the right edge and slice
        start = waveforms.shape[-1] - self.pre_whiten_size
        return waveforms[..., start:]
