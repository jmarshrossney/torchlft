from __future__ import annotations

import math

import torch

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

    # Takes a volume average of 2-dimensional shifts
    # For each lattice site (row), restores the geometry by representing
    # the row as a 2d array where the axis correspond to displacements
    # in the two lattice directions. Then takes a volume average by
    # averaging over all rows (lattice sites)
    return torch.stack(
        [
            row.view(L, T).roll(
                ((-i // T), (-i % L)),
                dims=(0, 1),
            )
            for i, row in enumerate(correlator_lexi)
        ],
        dim=0,
    ).mean(dim=0)
