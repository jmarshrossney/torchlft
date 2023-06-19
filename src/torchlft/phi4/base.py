"""
The advantage of subclass torch.nn.Module and registering distribution parameters
as buffers is that they can be moved to the correct device with Module.to(device).
This is done under the hood by PyTorch Lightning.
"""

from math import log, prod, pi as π

import torch
import torch.nn as nn
import torch.linalg as LA

from torchlft.nflow import NormalizingFlow  # typing only
from torchlft.utils.distribution import GaussianModule, DiagonalGaussianModule
from torchlft.utils.tensor import sum_except_batch
from torchlft.utils.lattice import laplacian_2d
from torchlft.utils.linalg import mv
from torchlft.typing import Tensor


class Phi4BaseAction(BaseAction):
    pass


# Thermodynamic limit β->0


class DiagonalGaussianAction(Phi4BaseAction):
    def __init__(self):
        super().__init__()

    def compute(self, ϕ: Tensor) -> Tensor:
        return sum_except_batch((ϕ**2) / 2)

    def gradient(self, ϕ: Tensor) -> Tensor:
        return ϕ.clone()

    def sample(
        self, sample_size: int, lattice_shape: tuple[int, int]
    ) -> Tensor:
        return self._reference.new_empty(
            (sample_size, *lattice_shape)
        ).normal_()

    def log_norm(self, lattice_shape: tuple[int, int]) -> float:
        return prod(lattice_shape) * 0.5 * log(pi)


# Gaussian limit λ->0
# NOTE: Will be really inefficient if the lattice shape changes often
# This is because constructing a torch.distributions.MultivariateNormal
# involves some linear algebra.
# TODO: replace this with a reparametrisation trick via rescaling and IFFT
class GaussianAction(Phi4BaseAction):
    def __init__(self, m_sq: float):
        super().__init__()

        # TODO: make trainable parameter, need reparametrisation via fft?
        self.register_buffer("m_sq", torch.tensor(m_sq, dtype=torch.float))

        self._lattice_shape = None
        self._distribution = None

    def get_distribution(self, lattice_shape: tuple[int, int]):
        # Have to re-generate distribution if lattice shape changes
        if not tuple(lattice_shape) == self._lattice_shape:
            assert len(lattice_shape) == 2
            assert lattice_shape[0] == lattice_shape[1]

            L, _ = lattice_shape
            action = (laplacian_2d(L) + self.m_sq + torch.eye(L)).type_as(
                self._reference
            )
            mean = self._reference.new_zeros(L**2)

            self._distribution = MultivariateNormal(
                loc=mean, precision_matrix=action
            )
            self._lattice_shape = lattice_shape

        # Very hacky way of moving distribution to the correct device
        if not self._distribution.loc.device == self._reference.device:
            self._distribution.loc = self._distribution.loc.to(
                self._reference.device
            )
            self._distribution._unbroadcasted_scale_tril = (
                self._distribution._unbroadcasted_scale_tril.to(
                    self._reference.device
                )
            )

        return self._distribution

    def compute(self, ϕ: Tensor) -> Tensor:
        _, *lattice_shape = ϕ.shape
        distribution = self.get_distribution(lattice_shape)
        log_density_gauss = distribution.log_prob(ϕ.flatten(start_dim=1))
        return log_density_gauss.negative()

    def _gradient_check(self, ϕ: Tensor) -> Tensor:
        _, *lattice_shape = ϕ.shape
        distribution = self.get_distribution(lattice_shape)
        K = distribution.precision_matrix
        return mv(K, ϕ.flatten(start_dim=1)).unflatten(1, self.lattice_shape)

    def gradient(self, ϕ: Tensor) -> Tensor:
        grad = ϕ.roll(-1, 1) + ϕ.roll(+1, 1) + ϕ.roll(-1, 2) + ϕ.roll(+1, 2)
        grad += (4 + self.m_sq) * ϕ
        return grad

    def sample(
        self, sample_size: int, lattice_shape: tuple[int, int]
    ) -> Tensor:
        distribution = self.get_distribution(lattice_shape)
        ϕ_flat = distribution.sample([sample_size])
        return ϕ_flat.unflatten(1, lattice_shape)

    def log_norm(self, lattice_shape: tuple[int, int]) -> float:
        distribution = self.get_distribution(lattice_shape)
        raise NotImplementedError  # TODO


class FlowedBaseAction(nn.Module):
    def __init__(self, base, flow: NormalizingFlow):
        super().__init__()
        self.base = base
        self.flow = flow

    def compute(self, ϕ_t: Tensor) -> Tensor:
        S_0 = self.base.compute(ϕ_t)
        ϕ_0, ldj = self.flow(ϕ_t)
        S_t = S_0 + ldj
        return S_t


# TODO: Ising limit? λ->inf. Or smooth bimodal prior in broken symmetry phase?
