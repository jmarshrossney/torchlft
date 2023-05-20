from __future__ import annotations

from functools import partial
from typing import NamedTuple, TYPE_CHECKING

import torch

from torchlft.utils.tensor import sum_except_batch

if TYPE_CHECKING:
    from torchlft.typing import *

__all__ = [
    "Couplings",
    "action",
    "Action",
]


class InvalidCouplings(Exception):
    pass


class Couplings(NamedTuple):
    r"""
    Coefficients for the three terms in the Phi^4 action.
    """

    β: float
    α: float
    λ: float

    @classmethod
    def particle_phys(cls, m_sq: float, λ: float) -> Couplings:
        r"""
        Standard Particle Physics parametrisation of Phi^4.
        """
        return cls(β=1, α=(4 + m_sq) / 2, λ=λ)

    @classmethod
    def ising(cls, β: float, λ: float) -> Couplings:
        r"""
        Ising-like parametrisation of Phi^4.
        """
        return cls(β=β, α=1 - 2 * λ, λ=λ)

    @classmethod
    def cond_mat(cls, κ: float, λ: float) -> Couplings:
        """
        Condensed matter parametrisation.
        """
        return cls(β=2 * κ, α=1 - 2 * λ, λ=λ)


_PARAMETRISATIONS = {
    ("beta", "alpha", "lamda"): Couplings,
    ("m_sq", "lamda"): Couplings.particle_phys,
    ("beta", "lamda"): Couplings.ising,
    ("kappa", "lamda"): Couplings.cond_mat,
    ("β", "α", "λ"): Couplings,
    ("m_sq", "λ"): Couplings.particle_phys,
    ("β", "λ"): Couplings.ising,
    ("κ", "λ"): Couplings.cond_mat,
}


def _parse_couplings(couplings: dict[str, float]) -> Couplings:
    try:
        parser = _PARAMETRISATIONS[tuple(couplings.keys())]
    except KeyError as exc:
        raise InvalidCouplings(
            f"{tuple(couplings.keys())} is not a known set of couplings"
        ) from exc
    return parser(**couplings)


def _action(
    sample: Tensor,
    couplings: Couplings,
) -> Tensor:
    """Computes the Phi^4 action for a single-component scalar field."""
    ϕ, β, α, λ = sample, *couplings

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


def get_action(**couplings: dict[str, float]) -> Callable[Tensor, Tensor]:
    r"""Phi^4 action for a single-component scalar field.

    .. math::

        S(\phi) = \sum_{x\in\Lambda} \left[
            -\kappa \sum_{\mu=1}^d \phi(x) \phi(x+e_\mu)
            + \alpha \phi(x)^2
            + \lambda \phi(x)^4
        \right]

    Args:
        sample
            Sample of field configurations with shape
            ``(batch_size, *lattice_shape)``
        couplings
            Either ``(m_sq, lam)`` or ``(beta, lam)``
    """
    return partial(_action, couplings=_parse_couplings(couplings))


class Action:
    """
    Class-based implementation of :func:`action`.
    """

    def __init__(self, **couplings: dict[str, float]) -> None:
        self._couplings = _parse_couplings(couplings)

    @property
    def couplings(self) -> Couplings:
        return self._couplings

    def __call__(self, sample: Tensor) -> Tensor:
        """Calls ``action`` with the sample provided."""
        return _action(sample, self._couplings)


def _local_action(
    phi: float,
    neighbours: list[float],
    couplings: Couplings,
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
