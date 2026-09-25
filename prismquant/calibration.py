"""Construct an anchor rotation from a calibration activation matrix."""
from __future__ import annotations
import torch
from .rotations import factor_from_vectors, RotationFactor


@torch.no_grad()
def fit_rotation(activations: torch.Tensor, rank: int | None = None,
                 group_size: int = 128) -> RotationFactor:
    """Fit uncentered principal directions and balance residual-coordinate energy.

    Input shape is (..., channels). This exact reference implementation forms a
    channels-by-channels second moment in fp64, so its memory is O(channels²).
    For large-scale streamed calibration, see the archived paper pipeline.
    """
    if activations.ndim < 2 or group_size < 1 or group_size & (group_size - 1):
        raise ValueError('Use an activation matrix and a positive power-of-two group size')
    n = activations.shape[-1]
    if n % group_size:
        raise ValueError('The channel dimension must be divisible by group_size')
    slots = n // group_size
    rank = slots if rank is None else rank
    if not isinstance(rank, int) or not 1 <= rank <= slots:
        raise ValueError(f'rank must be in [1, {slots}]')
    rows = activations.reshape(-1, n).float()
    if not rows.shape[0] or not torch.isfinite(rows).all():
        raise ValueError('Calibration activations must be nonempty and finite')
    moment = rows.double().T @ rows.double() / rows.shape[0]
    _, vectors = torch.linalg.eigh(moment)
    return factor_from_vectors(vectors[:, -rank:].flip(1).float(), rows, group_size)
