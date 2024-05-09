import torch


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

    def forward(self, waveforms: torch.Tensor, parameters):
        shifts = torch.rand(waveforms.size(0), device=waveforms.device)
        shifts = 2 * self.jitter * shifts - self.jitter
        shifts *= self.sample_rate
        waveforms = torch.roll(waveforms, shifts, dims=-1)
        return waveforms, parameters
