from math import prod

import torch

from torchlft.abc import BaseDensity, ScalarField
from torchlft.fields import CanonicalScalarField
from torchlft.typing import *
from torchlft.utils.distribution import expand_iid


class IsotropicGaussianBase(BaseDensity):
    def __init__(self, lattice_shape: torch.Size):
        super().__init__()
        self.lattice_shape = lattice_shape

        self.register_buffer("loc", torch.tensor([0.0]))
        self.register_buffer("scale", torch.tensor([1.0]))

        self._distribution = expand_iid(
            torch.distributions.Normal(loc=self.loc, scale=self.scale),
            extra_dims=lattice_shape,
        )

    def log_density(self, configs: ScalarField) -> Tensor:
        configs = configs.to_canonical()
        return self._distribution.log_prob(configs.tensor)

    def sample(self, batch_size: int) -> CanonicalScalarField:
        data = self._distribution.sample([batch_size])
        return CanonicalScalarField(data)


class FreeFieldBase(BaseDensity):
    def __init__(self, lattice_shape: torch.Size, m_sq: float):
        if len(lattice_shape) != 2:
            # TODO: support other dims
            raise ValueError(
                "Free field only supported for 2-dimensional lattice"
            )
        L1, L2 = lattice_shape
        if L1 != L2:
            raise ValueError("Only 2-d lattices of equal length are supported")

        # self._distribution = torch.distributions.MultivariateNormal(
        # loc=torch.zeros(
