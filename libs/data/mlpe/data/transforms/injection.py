from typing import TYPE_CHECKING

from ml4gw.transforms import RandomWaveformInjection

if TYPE_CHECKING:
    from ml4gw import gw


class WaveformInjector(RandomWaveformInjection):
    """Subclass of ml4gw's RandomWaveformInjection
    that injects waveforms at a fixed location
    into background kernels. The offset from the center
    is determine by `trigger_offset`
    """

    def forward(self, X: "gw.WaveformTensor") -> "gw.WaveformTensor":
        if self.training:
            # inject waveform with probability 1
            N = len(X)

            # infer kernel size of background
            kernel_size = X.shape[-1]

            # randomly sample waveforms
            waveforms, sampled_params = self.sample(N, device=X.device)

            # calculate the fixed location
            # where waveform T_c will placed
            center = (waveforms.shape[-1] // 2) + self.trigger_offset
            start = center - (kernel_size // 2)
            stop = center + (kernel_size // 2)

            waveforms = waveforms[:, :, start:stop]
            X += waveforms

            return X, sampled_params

        return X
