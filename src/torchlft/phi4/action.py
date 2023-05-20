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
            -\frac{\beta}{2} \sum_{\mu=1}^d \phi(x) \phi(x+e_\mu)
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


def _action_gradient(
    configs: Tensor,
    couplings: Couplings,
) -> Tensor:
    ϕ, β, α, λ = configs, *couplings

    grad = torch.zeros_like(ϕ)

    # Nearest neighbour interaction: +ve and -ve shifts
    for μ in range(1, ϕ.dim()):
        grad -= β * (ϕ.roll(-1, μ) + ϕ.roll(+1, μ))

    grad += 2 * α * ϕ

    grad += 4 * λ * ϕ.pow(3)

    return grad


def get_action_gradient(
    **couplings: dict[str, float]
) -> Callable[Tensor, Tensor]:
    return partial(_action_gradient, couplings=_parse_couplings(couplings))


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


class Action:
    """
    Class-based implementation of :func:`action`.
    """

    def __init__(self, **couplings: dict[str, float]):
        self._couplings = _parse_couplings(couplings)

    @property
    def couplings(self) -> Couplings:
        return self._couplings

    def compute(self, ϕ: Tensor) -> Tensor:
        """Calls ``action`` with the sample provided."""
        return _action(ϕ, self._couplings)

    def gradient(self, ϕ: Tensor) -> Tensor:
        return _action_gradient(ϕ, self._couplings)


class FlowedAction(nn.Module):
    def __init__(self, flow: NormalizingFlow, **couplings: dict[str, float]):
        super().__init__()
        self.flow = flow
        self._couplings = _parse_couplings(couplings)

    @property
    def couplings(self) -> Couplings:
        return self._couplings

    @torch.no_grad()
    def compute(self, ϕ_t: Tensor) -> Tensor:
        ϕ_0, ldj = self.flow(ϕ_t)
        return _action(ϕ_0, self._couplings) - ldj

    @torch.enable_grad()
    def gradient(self, ϕ_t: Tensor) -> Tensor:
        ϕ_t.requires_grad_(True)
        ϕ_0, ldj = self.flow(ϕ_t)
        S = _action(ϕ_0, self._couplings) - ldj
        (grad,) = torch.autograd.grad(
            outputs=S, inputs=ϕ_t, grad_outputs=torch.ones_like(S)
        )
        return grad
