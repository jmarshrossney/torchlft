import dataclasses
from typing import Optional, Union

import torch

import torchnf.distributions

import torchlft.phi_four.actions
import torchlft.utils


class FreeTheory(torch.distributions.MultivariateNormal):
    """
    A distribution representing a non-interacting scalar field.
    This is a subclass of torch.distributions.MultivariateNormal in which the
    covariance matrix is specified by the bare mass of the scalar field.

    Args:
        lattice_length
            Number of nodes on one side of the square 2-dimensional lattice.
        m_sq
            Bare mass, squared.
    """

    def __init__(self, lattice_length: int, m_sq: float) -> None:
        # TODO m_sq should be positive and nonzero
        super().__init__(
            loc=torch.zeros(lattice_length ** 2),
            precision_matrix=(
                torchlft.utils.laplacian_2d(lattice_length)
                + torch.eye(lattice_length ** 2).mul(m_sq)
            ),
        )
        self._lattice_length = lattice_length

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        """Flattens 2d configurations and calls superclass log_prob."""
        return super().log_prob(sample.flatten(start_dim=1))

    def rsample(self, sample_shape=torch.Size()) -> torch.Tensor:
        """Calls superclass rsample and restores 2d geometry."""
        return (
            super()
            .rsample(sample_shape)
            .view(*sample_shape, self._lattice_length, self._lattice_length)
        )


class SimpleGaussianPrior(torchnf.distributions.SimplePrior):
    """
    Gaussian distribution with unit diagonal covariance.
    """

    def __init__(
        self,
        lattice_shape: tuple[int, int],
        batch_size: int = 1,
    ):
        super().__init__(
            torch.distributions.Normal(0, 1),
            batch_size=batch_size,
            expand_shape=lattice_shape,
        )


class FreeTheoryPrior(torchnf.distributions.SimplePrior):
    """
    Multivariate Gaussian corresponding to free theory.

    .. seealso:: :class:`FreeTheory`
    """

    def __init__(
        self,
        lattice_length: int,
        m_sq: float,
        batch_size: int = 1,
    ):
        super().__init__(
            FreeTheory(lattice_length, m_sq),
            batch_size=batch_size,
        )


@dataclasses.dataclass
class PhiFourTargetStandard:
    m_sq: Union[float, torch.Tensor]
    lam: Union[float, torch.Tensor]

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        return torchlft.phi_four.actions.phi_four_action_standard(
            sample, self.m_sq, self.lam
        ).neg()


@dataclasses.dataclass
class PhiFourTargetIsing:
    beta: Union[float, torch.Tensor]
    lam: Union[float, torch.Tensor]

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        return torchlft.phi_four.actions.phi_four_action_ising(
            sample, self.beta, self.lam
        ).neg()


def PhiFourTarget(
    lam: Union[float, torch.Tensor],
    m_sq: Optional[Union[float, torch.Tensor]],
    beta: Optional[Union[float, torch.Tensor]],
) -> Union[PhiFourTargetStandard, PhiFourTargetIsing]:
    """
    Convenience function for selecting correct target distribution.
    """
    if m_sq is not None:
        assert beta is None, "provide either m_sq or beta, but not both"
        return PhiFourTargetStandard(m_sq, lam)
    elif beta is not None:
        assert m_sq is None, "provide either m_sq or beta, but not both"
        return PhiFourTargetIsing(beta, lam)
    else:
        raise ValueError("One of m_sq or beta should be provided")
