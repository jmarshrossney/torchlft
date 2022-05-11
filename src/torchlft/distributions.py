from __future__ import annotations

import torch

import torchlft.utils


# NOTE: this may be unnecessary - IterableDataset seems to do very little
class Prior(torch.utils.data.IterableDataset):
    """Wraps around torch.distributions.Distribution to make it iterable."""

    def __init__(
        self,
        distribution: torch.distributions.Distribution,
        batch_size: int = 1,
    ):
        super().__init__()
        assert isinstance(distribution, torch.distributions.Distribution)
        self._distribution = distribution
        self._batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self) -> tuple[torch.Tensor]:
        sample = self.sample()
        return sample, self.log_prob(sample)

    def sample(self) -> torch.Tensor:
        return self._distribution.sample([self.batch_size])

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        return (
            self._distribution.log_prob(sample).flatten(start_dim=1).sum(dim=1)
        )

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return self._distribution

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new: int) -> None:
        self._batch_size = new


class FreeScalarDistribution(torch.distributions.MultivariateNormal):
    r"""A distribution representing a non-interacting scalar field.

    This is a subclass of torch.distributions.MultivariateNormal in which the
    covariance matrix is specified by the bare mass of the scalar field.

    Parameters
    ----------
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
        return super().log_prob(sample.flatten(start_dim=-2))

    def rsample(self, sample_shape=torch.Size()) -> torch.Tensor:
        """Calls superclass rsample and restores 2d geometry."""
        return (
            super()
            .rsample(sample_shape)
            .view(*sample_shape, self._lattice_length, self._lattice_length)
        )
