"""Candidate peak-association score for comparisons and explicit task experiments."""

import torch

from .scoring import WhipScorer


@torch.no_grad()
def regional_peak_mask(prominence):
    """Positive local maxima of (B,T,3) bend-increase histories, on device.

    A peak must rise strictly from the previous sample and fall strictly after
    its entire equal-valued plateau. Choose the first sample of a flat top.
    Neither history endpoint is a peak. No smoothing, prominence threshold,
    minimum pulse width or floating-point equality tolerance is introduced.
    This offline rule uses future samples, possibly outside the event window;
    the scorer separately restricts the selected peak time to that window.
    """
    if (not isinstance(prominence, torch.Tensor) or prominence.ndim != 3
            or prominence.shape[-1] != 3 or prominence.shape[1] < 2
            or prominence.dtype not in (torch.float32, torch.float64)):
        raise ValueError('Prominence must be float32/float64 (batch,times,3), with at least two times')
    values = prominence.transpose(1, 2)
    count = values.shape[-1]
    index = torch.arange(count, device=values.device)
    # For each sample, locate the end of its maximal constant plateau.
    ends = torch.cat((values[..., :-1] != values[..., 1:],
                      torch.ones_like(values[..., :1], dtype=torch.bool)), dim=-1)
    end_index = torch.where(ends, index, count-1).flip(-1).cummin(-1).values.flip(-1)
    right = values.gather(-1, (end_index+1).clamp_max(count-1))
    rising = torch.cat((torch.zeros_like(values[..., :1], dtype=torch.bool),
                        values[..., 1:] > values[..., :-1]), dim=-1)
    return (rising & (values > 0) & torch.isfinite(values) & torch.isfinite(right)
            & (end_index < count-1) & (values > right)).transpose(1, 2)


class PeakWhipScorer(WhipScorer):
    """Same score and timing limits, restricted to regional bend-increase peaks.

    Amplitudes remain continuous, but peak association/appearance can change
    discretely. This does not track one bend, establish energy transfer or
    guarantee that peaks in different regions belong to the same pulse.
    """

    def _candidate_mask(self, prominence):
        return regional_peak_mask(prominence)
