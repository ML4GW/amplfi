import torch
import torch.nn.functional as F


class TimeTranslator(torch.nn.Module):
    """
    A torch.nn.Module that shifts waveforms in time

    Args:
        jitter:
            The maximum magnitude of time to shift waveforms in seconds.
            Waveforms will be shifted by a random
            amount between -jitter and jitter
        sample_rate:
            The rate at which waveforms are sampled in Hz
    """

    def __init__(self, jitter: float, sample_rate: float):
        super().__init__()
        self.jitter = jitter
        self.sample_rate = sample_rate
        self.pad = int(self.jitter * self.sample_rate)

    def forward(self, waveforms: torch.Tensor):
        # some array magic to shift waveforms
        # in time domain by different amounts
        shifts = torch.rand(waveforms.size(0), device=waveforms.device)
        shifts = 2 * self.jitter * shifts - self.jitter
        shifts *= self.sample_rate
        shifts = shifts.long()

        indices = (
            torch.arange(waveforms.size(-1), device=waveforms.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        indices = indices.expand(*waveforms.shape)
        indices = indices + self.pad
        shifted = indices + shifts.view(-1, 1, 1)

        waveforms = F.pad(
            waveforms, (self.pad, self.pad), mode="constant", value=0
        )
        waveforms = waveforms.gather(2, shifted)
        return waveforms
