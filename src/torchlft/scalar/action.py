from typing import TypeAlias

import torch

from torchlft.utils.lattice import laplacian
from torchlft.utils.linalg import dot, mv

Tensor: TypeAlias = torch.Tensor


class ActionV1:
    def __init__(self, lattice_length: int, m_sq: float, lattice_dim: int = 2):
        L, d = lattice_length, lattice_dim

        assert d in (1, 2)

        self.kernel = -laplacian(L, d) + m_sq * torch.eye(L**d)

        self.lattice_length = lattice_length
        self.m_sq = m_sq

    def __call__(self, φ: Tensor) -> Tensor:
        φ = φ.flatten(start_dim=1)
        K = self.kernel
        S = 0.5 * dot(φ, mv(K, φ))
        return S.unsqueeze(-1)

    def grad(self, φ: Tensor) -> Tensor:
        φ = φ.flatten(start_dim=1)
        K = self.kernel
        return mv(K, φ)


class ActionV2:
    def __init__(self, m_sq: float, lattice_dim: int = 2):
        self.m_sq = m_sq
        self.lattice_dim = lattice_dim

    def __call__(self, φ: Tensor) -> Tensor:
        lattice_dims = tuple(range(1, self.lattice_dim + 1))
        s = torch.zeros_like(φ)

        for μ in lattice_dims:  # TODO: accept different dims?
            s -= 0.5 * φ * φ.roll(-1, μ)
            s -= 0.5 * φ * φ.roll(+1, μ)

        s += 0.5 * (4 + self.m_sq) * φ**2

        return s.sum(dim=lattice_dims)

    def grad(self, φ: Tensor) -> Tensor:
        lattice_dims = tuple(range(1, self.lattice_dim + 1))
        dsdφ = torch.zeros_like(φ)

        for μ in lattice_dims:
            dsdφ -= φ.roll(-1, μ) + φ.roll(+1, μ)

        dsdφ += (4 + self.m_sq) * φ

        return dsdφ
