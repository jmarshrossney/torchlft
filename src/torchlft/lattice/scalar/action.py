from math import log, pi as π
from typing import TypeAlias

import torch
import torch.nn as nn

from torchlft.utils.lattice import laplacian
from torchlft.utils.linalg import dot, mv

Tensor: TypeAlias = torch.Tensor


class GaussianAction(nn.Module):
    def __init__(self, lattice_length: int, lattice_dim: int, m_sq: float):
        super().__init__()
        assert lattice_length % 2 == 0
        assert lattice_dim in (1, 2)
        assert m_sq > 0

        L, d = lattice_length, lattice_dim
        D = pow(L, d)

        K = -laplacian(L, d) + m_sq * torch.eye(D)
        Σ = torch.linalg.inv(K)
        C = torch.linalg.cholesky(Σ)

        log_abs_det_C = C.diag().log().sum()

        # check |Σ| = |C|^2
        assert torch.allclose(
            2 * log_abs_det_C,
            torch.linalg.slogdet(Σ)[1],
            atol=1e-5,
        )

        self.lattice_length = L
        self.lattice_dim = d
        self.lattice_size = D
        self.log_norm = 0.5 * D * log(2 * π) + log_abs_det_C

        self.register_buffer("kernel", K)
        self.register_buffer("covariance", Σ)
        self.register_buffer("cholesky", C)

    def forward(self, φ: Tensor) -> Tensor:
        K = self.kernel
        S = 0.5 * dot(φ, mv(K, φ))
        return S + self.log_norm

    def grad(self, φ: Tensor) -> Tensor:
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
