from collections.abc import Iterable
import dataclasses
from typing import NamedTuple, Union

from jsonargparse.typing import PositiveInt
import torch

__all__ = [
    "PhiFourCoefficients",
    "phi_four_action",
    "PhiFourAction",
]


class InvalidParametrisation(Exception):
    pass


class PhiFourCoefficients(NamedTuple):
    r"""
    Coefficients for the three terms in the Phi^4 action.
    """

    beta: float
    alpha: float
    lamda: float

    @classmethod
    def particle_phys(cls, m_sq: float, lamda: float) -> "PhiFourCoefficients":
        r"""
        Standard Particle Physics parametrisation of Phi^4.
        """
        return cls(beta=1, alpha=(4 + m_sq) / 2, lamda=lamda)

    @classmethod
    def ising(cls, beta: float, lamda: float) -> "PhiFourCoefficients":
        r"""
        Ising-like parametrisation of Phi^4.
        """
        return cls(beta=beta, alpha=1 - 2 * lamda, lamda=lamda)

    @classmethod
    def cond_mat(cls, kappa: float, lamda: float) -> "PhiFourCoefficients":
        """
        Condensed matter parametrisation.
        """
        return cls(beta=2 * kappa, alpha=1 - 2 * lamda, lamda=lamda)


_PARAMETRISATIONS = {
    ("beta", "alpha", "lamda"): PhiFourCoefficients,
    ("m_sq", "lamda"): PhiFourCoefficients.particle_phys,
    ("beta", "lamda"): PhiFourCoefficients.ising,
    ("kappa", "lamda"): PhiFourCoefficients.cond_mat,
}


def _parse_couplings(couplings: dict[str, float]) -> tuple[float]:
    try:
        parser = _PARAMETRISATIONS[tuple(couplings.keys())]
    except KeyError as exc:
        raise InvalidParametrisation(
            f"{tuple(couplings.keys())} is not a known set of couplings"
        ) from exc
    return parser(**couplings)


def _action(
    sample: torch.Tensor,
    coeffs: PhiFourCoefficients,
) -> torch.Tensor:
    """Computes the Phi^4 action for a single-component scalar field."""
    beta, alpha, lamda = coeffs
    phi = sample
    action_density = torch.zeros_like(phi)

    # Nearest neighbour interaction
    for dim in range(1, phi.dim()):
        action_density.sub_(phi.mul(phi.roll(-1, dim)).mul(beta))

    # phi^2 term
    phi_sq = phi.pow(2)
    action_density.add_(phi_sq.mul(alpha))

    # phi^4 term
    action_density.add_(phi_sq.pow(2).mul(lamda))

    # Sum over lattice sites
    return action_density.flatten(start_dim=1).sum(dim=1)


def _local_action(
    phi: float,
    neighbours: list[float],
    coeffs: PhiFourCoefficients,
) -> float:
    """
    Computes the local Phi^4 action for a single-component scalar field.
    """
    beta, alpha, lamda = coeffs
    return (
        -0.5 * beta * phi * sum(neighbours)
        + alpha * phi**2
        + lamda * phi**4
    )


def phi_four_action(
    sample: torch.Tensor, **couplings: dict[str, float]
) -> torch.Tensor:
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
    coeffs = _parse_couplings(couplings)
    return _action(sample, coeffs)


def phi_four_action_local(
    phi: float, neighbours: list[float], **couplings: dict[str, float]
) -> float:
    coeffs = _parse_couplings(couplings)
    return _local_action(phi, neighbours, coeffs)


class PhiFourAction:
    """
    Class-based implementation of :func:`phi_four_action`.

    This class can serve as a 'target distribution' in e.g. a
    Normalizing Flow, through the ``log_prob`` method. See
    :py:class:`torchnf.abc.TargetDistribution`.
    """

    def __init__(self, **couplings: dict[str, float]) -> None:
        self._coeffs = _parse_couplings(couplings)

    @property
    def coefficients(self) -> PhiFourCoefficients:
        return self._coeffs

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        """Calls ``phi_four_action`` with the sample provided."""
        return _action(sample, self._coeffs)

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        """Returns the negated action."""
        return self(sample).neg()
