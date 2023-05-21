from __future__ import annotations

import os
import pathlib
from collections.abc import Iterator
from math import exp
from random import random
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.utils.tensorboard as tensorboard
import tqdm

from torchlft.actions import (
    phi_four_action,
    phi_four_action_local,
)
from torchlft.utils.lattice import build_neighbour_list
from torchlft.utils.distribution import Gaussian, DiagonalGaussian
from torchlft.utils.tensor import sum_except_batch

Tensor = torch.Tensor


class Hamiltonian_(nn.Module):
    def __init__(self, action, auxiliary):
        super().__init__()
        self.action = action
        self.auxiliary = auxiliary

    def kinetic(self, η: Tensor) -> Tensor:
        return self.auxiliary.compute(η)

    def potential(self, ϕ: Tensor) -> Tensor:
        return self.action.compute(ϕ)

    def grad_kinetic(self, η: Tensor) -> Tensor:
        return self.auxiliary.gradient(η)

    def grad_potential(self, ϕ: Tensor) -> Tensor:
        return self.action.gradient(ϕ)

    def compute(self, ϕ: Tensor, η: Tensor) -> Tensor:
        return self.kinetic(η) + self.potential(ϕ)


class Hamiltonian(nn.Module):
    def __init__(self, action):
        super().__init__()
        self.action = action

    def compute(self, ϕ: Tensor, η: Tensor) -> Tensor:
        return self.kinetic(η) + self.potential(ϕ)

    def sample_momenta(self, ϕ: Tensor) -> Tensor:
        ...

    def kinetic(self, η: Tensor) -> Tensor:
        ...

    def potential(self, ϕ: Tensor) -> Tensor:
        return self.action.compute(ϕ)

    def grad_kinetic(self, η: Tensor) -> Tensor:
        ...

    def grad_potential(self, ϕ: Tensor) -> Tensor:
        return self.action.gradient(ϕ)


class HamiltonianGaussianMomenta(Hamiltonian):
    def sample_momenta(self, ϕ: Tensor) -> Tensor:
        return torch.empty_like(ϕ).normal_()

    def kinetic(self, η: Tensor) -> Tensor:
        return sum_except_batch(η**2) / 2

    def grad_kinetic(self, η: Tensor) -> Tensor:
        return η.clone()


"""
    if mass_matrix is None:
        self.auxiliary = DiagonalGaussian(lattice_shape)
    else:
        L, T = lattice_shape
        assert mass_matrix.shape == (L * T, L * T)
        self.auxiliary = Gaussian(
            mean=torch.zeros(L * T), precision=mass_matrix
        )
"""
