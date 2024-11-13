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


class TimeAndPhaseShifter(torch.nn.Module):
    """
    A ``torch.nn.Module`` that augments frequency-domain waveforms
    by time and phase shifts. The output is computed as

    .. math::
        h(f) = \exp(2\pi i f t_c) * \exp(2 i \phi_c) * h(f)  # noqa: W605

    Args:
        time_jitter:
            The maximum magnitude of time to shift waveforms in seconds.
    """

    def __init__(self, time_jitter: float = 1.0):
        super().__init__()
        self.time_jitter = time_jitter

    def forward(self, waveforms: torch.Tensor, frequency: torch.Tensor):
        """
        Returns the augmented waveform array. Both arguments assumed to
        be on same device.

        Args:
            waveforms:
                The waveform array, assumed to be shape
                (batch, num_polarization, num_freqs).
            frequency:
                The frequencies for the waveform, assumed to be
                1-D tensor of shape num_freqs
        """
        if waveforms.shape[-1] != frequency.shape[-1]:
            raise RuntimeError(
                "The dimension of h(f) array must match number of frequencies"
            )
        batch_size, num_polarization, _ = waveforms.shape
        time_shifts = self.time_jitter * torch.rand(
            batch_size, device=waveforms.device
        )
        # random time shift up to time_jitter
        phase_shift = torch.exp(
            1j * 2 * torch.pi * torch.outer(time_shifts, frequency)
        )
        # random phase angle up to two pi
        phase_angle = (
            2 * torch.pi * torch.rand(batch_size, device=waveforms.device)
        )
        phase_shift *= torch.exp(
            1j * 2 * torch.outer(phase_angle, torch.ones_like(frequency))
        )
        # repeat phase shifts across polarizations
        phase_shift = phase_shift.unsqueeze(1).repeat(1, num_polarization, 1)
        return waveforms * phase_shift
