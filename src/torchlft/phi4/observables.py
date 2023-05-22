from __future__ import annotations

import math

import torch

from torchlft.utils.lattice import correlator_restore_geom

Tensor = torch.Tensor


def one_point_susceptibility(sample: Tensor, bessel: bool = False) -> Tensor:
    _, L, T = sample.shape
    M = sample.sum(dim=(1, 2))
    χ = M.var(unbiased=bessel) / (L * T)
    return χ


def two_point_correlator(sample: Tensor, bessel: bool = False) -> Tensor:
    """

    Note: lattice must be 2d
    """
    _, L, T = sample.shape

    # Compute correlation matrix for configs viewed as a 1d vector
    covariance_lexi = torch.cov(
        sample.flatten(start_dim=1).transpose(),
        correction=1 if bessel else 0,
    )

    assert covariance_lexi.shape == (L * T, L * T)

    # Rescale so that the diagonal elements are 1 on average
    correlator_lexi = covariance_lexi / covariance_lexi.diagonal().mean()

    return correlator_restore_geom(correlator_lexi, (L, T))
