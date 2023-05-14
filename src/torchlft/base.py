"""

The advantage of subclass torch.nn.Module and registering distribution parameters
as buffers is that they can be moved to the correct device with Module.to(device).
This is done under the hood by PyTorch Lightning.
"""
from math import prod, pi as π

import torch
import torch.linalg as LA

from torchlft.abc import BaseDensity, ScalarField
from torchlft.constraints import real, periodic, unit_vector
from torchlft.fields import (
    CanonicalScalarField,
    CanonicalPeriodicScalarField,
    CanonicalClassicalSpinField,
)
from torchlft.typing import *
from torchlft.utils.distribution import expand_iid
from torchlft.utils.tensor import sum_except_batch
from torchlft.utils.lattice import laplacian_2d


class IsotropicGaussianBase(BaseDensity):
    domain = real

    def __init__(self, lattice_shape: torch.Size):
        super().__init__()
        self.lattice_shape = lattice_shape

        self.register_buffer("loc", torch.tensor([0.0]))
        self.register_buffer("scale", torch.tensor([1.0]))

        self._distribution = expand_iid(
            torch.distributions.Normal(loc=self.loc, scale=self.scale),
            extra_dims=lattice_shape,
        )

    def action(self, configs: ScalarField) -> Tensor:
        configs = configs.to_canonical()
        log_density = self._distribution.log_prob(configs.tensor)
        return log_density.negative()

    def sample(self, batch_size: int) -> CanonicalScalarField:
        data = self._distribution.sample([batch_size])
        return CanonicalScalarField(data)


class FreeScalarBase(BaseDensity):
    domain = real

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

    def action(self, configs: ScalarField) -> Tensor:
        configs = configs.to_canonical()
        log_density = self._distribution.log_prob(
            configs.tensor.flatten(start_dim=1)
        )
        return log_density.negative()

    def sample(self, batch_size: int) -> CanonicalScalarField:
        data = self._distribution.sample([batch_size])
        return CanonicalScalarField(data.view(batch_size, *self.lattice_shape))


class UniformAnglesBase(BaseDensity):
    domain = periodic

    def __init__(self, lattice_shape: torch.Size):
        self.lattice_shape = lattice_shape

        self.register_buffer("low", torch.tensor([0.0]))
        self.register_buffer("high", torch.tensor([2 * π]))

        self._distribution = expand_iid(
            torch.distributions.Uniform(low=self.low, high=self.high),
            extra_dims=lattice_shape,
        )

    def action(self, configs: PeriodicScalarField) -> Tensor:
        configs = configs.to_canonical()
        log_density = self._distribution.log_prob(configs.tensor)
        return log_density.negative()

    def sample(self, batch_size: int) -> CanonicalPeriodicScalarField:
        data = self._distribution.sample([batch_size])
        return CanonicalPeriodicScalarField(data)


class IsotropicSphericalBase(BaseDensity):
    domain = unit_vector

    def __init__(self, lattice_shape: torch.Size, sphere_dim: int):
        self.lattice_shape = lattice_shape
        self.sphere_dim = sphere_dim

        self.register_buffer("loc", torch.tensor([0.0]))
        self.register_buffer("scale", torch.tensor([1.0]))

        self._distribution = expand_iid(
            torch.distributions.Normal(loc=self.loc, scale=self.scale),
            extra_dims=(*lattice_shape, sphere_dim + 1),
        )

    def action(self, configs: ScalarField) -> Tensor:
        return torch.zeros(
            configs.tensor.shape[0],
            device=configs.tensor.device,
            dtype=configs.tensor.dtype,
        )

    def sample(self, batch_size: int) -> CanonicalClassicalSpinField:
        data = self._distribution.sample([batch_size])
        data = data / LA.vector_norm(data, dim=-1)
        return CanonicalClassicalSpinField(data)
