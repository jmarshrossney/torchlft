from typing import TypeAlias

import torch

Tensor: TypeAlias = torch.Tensor

def susceptibility(φ: Tensor, bessel: bool = False) -> Tensor:
    _, L, T, _ = φ.shape
    M = φ.sum(dim=(1, 2))
    χ = M.var(unbiased=bessel) / (L * T)
    return χ


def restore_geom(lexi: Tensor, lattice: tuple[int, int]) -> Tensor:
    L, T = lattice
    return torch.stack(
        [
            row.view(L, T).roll(
                ((-i // T), (-i % L)),
                dims=(0, 1),
            )
            for i, row in enumerate(lexi)
        ],
        dim=0,
    ).mean(dim=0)

def two_point_correlator(φ: Tensor, bessel: bool = False) -> Tensor:
    _, L, T, _ = φ.shape

    covariance_lexi = torch.cov(
        φ.flatten(start_dim=1).transpose(0, 1),
        correction=1 if bessel else 0,
    )

    assert covariance_lexi.shape == (L * T, L * T)

    # Rescale so that the diagonal elements are on av
    #correlator_lexi = covariance_lexi / covariance_lexi.diagonal().mean()

    return restore_geom(covariance_lexi, (L, T))
