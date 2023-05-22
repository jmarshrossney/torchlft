from __future__ import annotations

from math import pi as π

import torch

from torchlft.utils.lattice import correlator_restore_geom
from torchlft.utils.linalg import outer, dot
from torchlft.utils.tensor import mod_2pi, sum_except_batch


def two_point_correlator(z: Tensor) -> Tensor:
    _, L, T, N = z.shape

    z_lexi = z.flatten(start_dim=1, end_dim=-2)

    zx_zy = torch.einsum("bxi,byi->bxy", z_lexi.conj(), z_lexi)

    mod_sq = zx_zy.real**2 + zx_zy.imag**2

    correlator_lexi = torch.mean(mod_sq, dim=0)

    return correlator_restore_geom(correlator_lexi, (L, T))


def two_point_correlator_v2(z: Tensor) -> Tensor:
    B, L, T, N = z.shape

    P = outer(z, z.conj())

    P = P.flatten(start_dim=1, end_dim=2)

    # < Tr(Px Py) >
    correlator_lexi = torch.einsum("bxij,byji->xy", P, P) / B

    return correlator_restore_geom(correlator_lexi, (L, T))


def topological_charge_geometric(z: Tensor) -> Tensor:
    P1 = outer(z, z.conj())
    P2 = P1.roll(-1, 1)
    P3 = P2.roll(-1, 2)
    P4 = P3.roll(+1, 1)

    tr_P1P2P3 = torch.einsum(
        "bxyij,bxyjk,bxyki->bxy",
        P1,
        P2,
        P3,
    )
    tr_P1P3P4 = torch.einsum(
        "bxyij,bxyjk,bxyki->bxy",
        P1,
        P3,
        P4,
    )

    area_enclosed = torch.einsum(
        "bxy->b",
        2 * torch.atan2(tr_P1P2P3.imag, tr_P1P2P3.real)
        + 2 * torch.atan2(tr_P1P3P4.imag, tr_P1P3P4.real),
    )

    return area_enclosed / (4 * π)


def topological_charge_v2(z: Tensor) -> Tensor:
    θ1 = dot(z.conj(), z.roll(-1, 1)).angle()
    θ2 = dot(z.conj(), z.roll(-1, 2)).angle()

    q = θ1 + θ2.roll(-1, 1) - θ1.roll(-1, 2) - θ2

    q = mod_2pi(q + π) - π  # -π < q < π
    
    return sum_except_batch(q) / (2 * π)

def topological_charge_v3(A: Tensor) -> Tensor:
    A1, A2 = A.split(1, dim=-1)
    q = A1 + A2.roll(-1, 1) - A1.roll(-1, 2) - A2

    q = mod_2pi(q + π) - π  # -π < q < π
    
    return sum_except_batch(q) / (2 * π)



