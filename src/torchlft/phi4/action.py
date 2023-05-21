from __future__ import annotations

from functools import partial
from typing import NamedTuple, TYPE_CHECKING

import torch
import torch.nn as nn

from torchlft.utils.tensor import sum_except_batch

if TYPE_CHECKING:
    from torchlft.typing import *


class InvalidCouplings(Exception):
    pass


class Phi4Couplings(NamedTuple):
    r"""
    Coefficients for the three terms in the Phi^4 action.
    """

    β: float
    α: float
    λ: float

    @classmethod
    def particle_phys(cls, m_sq: float, λ: float) -> Phi4Couplings:
        r"""
        Standard Particle Physics parametrisation of Phi^4.
        """
        return cls(β=1, α=(4 + m_sq) / 2, λ=λ)

    @classmethod
    def ising(cls, β: float, λ: float) -> Phi4Couplings:
        r"""
        Ising-like parametrisation of Phi^4.
        """
        return cls(β=β, α=1 - 2 * λ, λ=λ)

    @classmethod
    def cond_mat(cls, κ: float, λ: float) -> Phi4Couplings:
        """
        Condensed matter parametrisation.
        """
        return cls(β=2 * κ, α=1 - 2 * λ, λ=λ)


_PARAMETRISATIONS = {
    ("beta", "alpha", "lamda"): Phi4Couplings,
    ("m_sq", "lamda"): Phi4Couplings.particle_phys,
    ("beta", "lamda"): Phi4Couplings.ising,
    ("kappa", "lamda"): Phi4Couplings.cond_mat,
    ("β", "α", "λ"): Phi4Couplings,
    ("m_sq", "λ"): Phi4Couplings.particle_phys,
    ("β", "λ"): Phi4Couplings.ising,
    ("κ", "λ"): Phi4Couplings.cond_mat,
}


def _parse_couplings(couplings: dict[str, float]) -> Phi4Couplings:
    try:
        parser = _PARAMETRISATIONS[tuple(couplings.keys())]
    except KeyError as exc:
        raise InvalidCouplings(
            f"{tuple(couplings.keys())} is not a known set of couplings"
        ) from exc
    return parser(**couplings)


class Phi4Action:
    def __init__(self, **couplings: dict[str, float]):
        self._couplings = _parse_couplings(couplings)

    @property
    def couplings(self) -> Phi4Couplings:
        return self._couplings

    def compute(self, ϕ: Tensor) -> Tensor:
        β, α, λ = self.couplings

        s = torch.zeros_like(ϕ)

        # Nearest neighbour interaction
        for μ in range(1, ϕ.dim()):
            s -= β * (ϕ * ϕ.roll(-1, μ))

        # phi^2 term
        ϕ_sq = ϕ**2
        s += α * ϕ_sq

        # phi^4 term
        s += λ * ϕ_sq**2

        # Sum over lattice sites
        return sum_except_batch(s)

    def gradient(self, ϕ: Tensor) -> Tensor:
        β, α, λ = self.couplings

        grad = torch.zeros_like(ϕ)

        # Nearest neighbour interaction: +ve and -ve shifts
        for μ in range(1, ϕ.dim()):
            grad -= β * (ϕ.roll(-1, μ) + ϕ.roll(+1, μ))

        grad += 2 * α * ϕ

        grad += 4 * λ * ϕ.pow(3)

        return grad


class FlowedPhi4Action(nn.Module):
    def __init__(self, flow: NormalizingFlow, **couplings: dict[str, float]):
        super().__init__()
        self.flow = flow
        self.target = Phi4Action(**couplings)

    def _compute(self, ϕ_t: Tensor) -> Tensor:
        ϕ_0, ldj = self.flow(ϕ_t)
        S_0 = self.target.compute(ϕ_0)
        S_t = S_0 - ldj
        return S_t

    @torch.no_grad()
    def compute(self, ϕ_t: Tensor) -> Tensor:
        return self._compute(ϕ_t)

    @torch.enable_grad()
    def gradient(self, ϕ_t: Tensor) -> Tensor:
        ϕ_t.requires_grad_(True)
        S_t = self._compute(ϕ_t)
        (grad,) = torch.autograd.grad(
            outputs=S_t, inputs=ϕ_t, grad_outputs=torch.ones_like(S_t)
        )
        return grad


def _local_action(
    phi: float,
    neighbours: list[float],
    couplings: Phi4Couplings,
) -> float:
    """
    Computes the local Phi^4 action for a single-component scalar field.
    """
    ϕi, ϕj, β, α, λ = phi, neighbours, *couplings
    return (-β / 2) * ϕi * sum(ϕj) + α * ϕi**2 + λ * ϕi**4


def get_local_action(
    phi: float, neighbours: list[float], **couplings: dict[str, float]
) -> Callable[[float, list[float]], float]:
    return partial(_local_action, couplings=_parse_couplings(couplings))
