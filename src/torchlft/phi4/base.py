"""
The advantage of subclass torch.nn.Module and registering distribution parameters
as buffers is that they can be moved to the correct device with Module.to(device).
This is done under the hood by PyTorch Lightning.
"""

from abc import ABC, ABCMeta, abstractmethod
from math import prod, pi as π

import torch
import torch.nn as nn
import torch.linalg as LA

from torchlft.nflow import NormalizingFlow  # typing only
from torchlft.utils.distribution import GaussianModule, DiagonalGaussianModule
from torchlft.utils.tensor import sum_except_batch
from torchlft.utils.lattice import laplacian_2d
from torchlft.utils.linalg import mv
from torchlft.typing import Tensor


class Phi4BaseAction(ABC):
    @abstractmethod
    def compute(self, ϕ: Tensor) -> Tensor:
        ...

    @abstractmethod
    def gradient(self, ϕ: Tensor) -> Tensor:
        ...

    @abstractmethod
    def sample(self, n: int) -> Tensor:
        ...


# Thermodynamic limit β->0
class DiagonalGaussianAction(DiagonalGaussianModule, Phi4BaseAction):
    def __init__(self, lattice_shape: tuple[int, ...]):
        super().__init__(shape=lattice_shape)
        self.lattice_shape = lattice_shape

    def compute(self, ϕ: Tensor) -> Tensor:
        return self.distribution.log_prob(ϕ).negative()

    def gradient(self, ϕ: Tensor) -> Tensor:
        return ϕ / self.distribution.scale.pow(2)

    def sample(self, n: int) -> Tensor:
        return self.distribution.sample([n])


# Gaussian limit λ->0
class GaussianAction(GaussianModule, Phi4BaseAction):
    def __init__(self, lattice_shape: tuple[int, int], m_sq: float):
        if len(lattice_shape) != 2:
            # TODO: support other dims
            raise ValueError(
                "Free field only supported for 2-dimensional lattice"
            )
        L, T = lattice_shape
        if L != T:
            raise ValueError("Only 2-d lattices of equal length are supported")

        # Currently this only works for square lattices
        precision = laplacian_2d(L) + m_sq * torch.eye(L * T)
        mean = torch.zeros(L * T)

        super().__init__(mean, precision=precision)

        self.lattice_shape = lattice_shape

    def compute(self, ϕ: Tensor) -> Tensor:
        log_density_gauss = self.distribution.log_prob(ϕ.flatten(start_dim=1))
        return log_density_gauss.negative()

    def gradient(self, ϕ: Tensor) -> Tensor:
        K = self.distribution.precision_matrix
        return mv(K, ϕ)

    def gradient_(self, ϕ: Tensor) -> Tensor:
        grad = ϕ.roll(-1, 1) + ϕ.roll(+1, 1) + ϕ.roll(-1, 2) + ϕ.roll(+1, 2)
        grad += (4 + self.m_sq) * ϕ
        return grad

    def sample(self, n: int) -> Tensor:
        ϕ_flat = self.distribution.sample([n])
        return ϕ_flat.unflatten(1, self.lattice_shape)


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
