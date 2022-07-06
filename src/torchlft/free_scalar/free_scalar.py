import dataclasses
from typing import Optional, Union

import torch

import torchnf.distributions

import torchlft.phi_four.actions
import torchlft.utils


class FreeScalarDistribution(torch.distributions.MultivariateNormal):
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
            loc=torch.zeros(lattice_length**2),
            precision_matrix=(
                torchlft.utils.laplacian_2d(lattice_length)
                + torch.eye(lattice_length**2).mul(m_sq)
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
