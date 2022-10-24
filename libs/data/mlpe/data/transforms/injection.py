from typing import TYPE_CHECKING, List, Optional

from ml4gw.transforms import RandomWaveformInjection

if TYPE_CHECKING:
    import numpy as np

    from ml4gw import gw
    from ml4gw.transforms.injection import SourceParameter


class FixedLocationWaveformInjection(RandomWaveformInjection):
    def __init__(
        self,
        sample_rate: float,
        ifos: List[str],
        dec: "SourceParameter",
        psi: "SourceParameter",
        phi: "SourceParameter",
        intrinsic_parameters: Optional["np.ndarray"] = None,
        highpass: Optional[float] = None,
        trigger_offset: float = 0,
        **polarizations: "np.ndarray",
    ):

        super().__init__(
            sample_rate,
            ifos,
            dec,
            psi,
            phi,
            intrinsic_parameters=intrinsic_parameters,
            highpass=highpass,
            trigger_offset=trigger_offset,
            **polarizations,
        )

    def forward(self, X: "gw.WaveformTensor") -> "gw.WaveformTensor":
        if self.training:
            # inject waveform with probability 1
            N = len(X)

            # infer kernel size of background
            kernel_size = X.shape[-1]

            # randomly sample waveforms
            waveforms, sampled_params = self.sample(N)

            # calculate the fixed location
            # where waveform tc will placed
            center = (waveforms.shape[-1] // 2) + self.trigger_offset
            start = center - (kernel_size // 2)
            stop = center + (kernel_size // 2)

            waveforms = waveforms[:, :, start:stop]
            X += waveforms

            return X, sampled_params
