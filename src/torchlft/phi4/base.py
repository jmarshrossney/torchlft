"""

The advantage of subclass torch.nn.Module and registering distribution parameters
as buffers is that they can be moved to the correct device with Module.to(device).
This is done under the hood by PyTorch Lightning.
"""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from math import prod, pi as π
from typing import TYPE_CHECKING

import torch
import torch.linalg as LA

from torchlft.abc import BaseAction
from torchlft.fields import CanonicalScalarField, ScalarField
from torchlft.utils.distribution import expand_iid
from torchlft.utils.tensor import sum_except_batch
from torchlft.utils.lattice import laplacian_2d

if TYPE_CHECKING:
    from torchlft.typing import *


# Thermodynamic limit β->0
class ActionThermodynamicLimit(BaseAction):
    def __init__(self, lattice_shape: torch.Size):
        super().__init__()
        self.lattice_shape = lattice_shape

        self.register_buffer("loc", torch.tensor([0.0]))
        self.register_buffer("scale", torch.tensor([1.0]))

        self._distribution = expand_iid(
            torch.distributions.Normal(loc=self.loc, scale=self.scale),
            extra_dims=lattice_shape,
        )

    def compute(self, ϕ: Tensor) -> Tensor:
        log_density_gauss = self._distribution.log_prob(ϕ)
        return log_density_gauss.negative()

    def gradient(self, ϕ: Tensor) -> Tensor:
        return ϕ / self.scale.pow(2)

    def sample(self, n: int) -> Tensor:
        return self._distribution.sample([n])


# Gaussian limit λ->0
class ActionGaussianLimit(BaseAction):
    def __init__(self, lattice_shape: torch.Size, m_sq: float):
        if len(lattice_shape) != 2:
            # TODO: support other dims
            raise ValueError(
                "Free field only supported for 2-dimensional lattice"
            )
        L, T = lattice_shape
        if L != T:
            raise ValueError("Only 2-d lattices of equal length are supported")

        self.lattice_shape = lattice_shape

        # Currently this only works for square lattices
        inverse_covariance = laplacian_2d(L) + m_sq * torch.eye(L * T)

        self.register_buffer("loc", torch.zeros(L * T))
        self.register_buffer("inverse_covariance", inverse_covariance)

        self._distribution = torch.distributions.MultivariateNormal(
            loc=self.log,
            precision_matrix=self.inverse_covariance,
        )

    def compute(self, ϕ: Tensor) -> Tensor:
        log_density_gauss = self._distribution.log_prob(ϕ.flatten(start_dim=1))
        return log_density_gauss.negative()

    def gradient(self, ϕ: Tensor) -> Tensor:
        raise NotImplementedError  # TODO

    def sample(self, n: int) -> Tensor:
        ϕ_flat = self._distribution.sample([n])
        return ϕ_flat.unflatten(1, self.lattice_shape)


# TODO: Ising limit? λ->inf
