from math import log, pi as π
from typing import NamedTuple, Self, TypeAlias

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
        self.m_sq = m_sq
        self.log_norm = 0.5 * D * log(2 * π) + log_abs_det_C

        self.register_buffer("kernel", K)
        self.register_buffer("covariance", Σ)
        self.register_buffer("cholesky", C)

    def extra_repr(self):
        return f"lattice_length={self.lattice_length}, lattice_dim={self.lattice_dim}, m_sq={self.m_sq}"

    def forward(self, φ: Tensor) -> Tensor:
        K = self.kernel
        S = 0.5 * dot(φ, mv(K, φ))
        return S + self.log_norm

    def grad(self, φ: Tensor) -> Tensor:
        K = self.kernel
        return mv(K, φ)


class InvalidCouplings(Exception):
    pass


#TODO check valid ?
class Phi4Couplings(NamedTuple):
    r"""
    Coefficients for the three terms in the Phi^4 action.
    """

    β: float
    α: float
    λ: float

    @classmethod
    def particle_phys(cls, m_sq: float, λ: float) -> Self:
        """
        Standard Particle Physics parametrisation of Phi^4.
        """
        return cls(β=1, α=(4 + m_sq) / 2, λ=λ)

    @classmethod
    def ising(cls, β: float, λ: float) -> Self:
        """
        Ising-like parametrisation of Phi^4.
        """
        return cls(β=β, α=1 - 2 * λ, λ=λ)

    @classmethod
    def cond_mat(cls, κ: float, λ: float) -> Self:
        """
        Condensed matter parametrisation.
        """
        return cls(β=2 * κ, α=1 - 2 * λ, λ=λ)


_PARAMETRISATIONS = {
    ("m_sq", "lambda"): Phi4Couplings.particle_phys,
    ("m_sq", "λ"): Phi4Couplings.particle_phys,
    ("beta", "lambda"): Phi4Couplings.ising,
    ("β", "λ"): Phi4Couplings.ising,
    ("kappa", "lambda"): Phi4Couplings.cond_mat,
    ("κ", "λ"): Phi4Couplings.cond_mat,
}


def _parse_couplings(couplings: dict[str, float]) -> Phi4Couplings:
    try:
        parser = _PARAMETRISATIONS[tuple(couplings.keys())]
    except KeyError as exc:
        raise InvalidCouplings(
            f"{tuple(couplings.keys())} is not a known set of couplings"
        ) from exc
    return parser(**couplings)


class Phi4Action(nn.Module):
    def __init__(self, **couplings: dict[str, float]):
        super().__init__()
        self._original_couplings = couplings
        self._parsed_couplings = _parse_couplings(couplings)

    def extra_repr(self):
        return str(self._original_couplings)

    @property
    def couplings(self) -> Phi4Couplings:
        return self._original_couplings

    def forward(self, ϕ: Tensor) -> Tensor:
        β, α, λ = self._parsed_couplings
        lattice_dims = (1, 2)

        s = torch.zeros_like(ϕ)

        # Nearest neighbour interaction
        for μ in lattice_dims:
            s -= β * (ϕ * ϕ.roll(-1, μ))

        # phi^2 term
        ϕ_sq = ϕ**2
        s += α * ϕ_sq

        # phi^4 term
        s += λ * ϕ_sq**2

        # Sum over lattice sites
        return s.sum(dim=lattice_dims)

    def grad(self, ϕ: Tensor) -> Tensor:
        β, α, λ = self._parsed_couplings
        lattice_dims = (1, 2)

        dsdφ = torch.zeros_like(φ)

        # Nearest neighbour interaction: +ve and -ve shifts
        for μ in lattice_dims:
            dsdφ -= β * (ϕ.roll(-1, μ) + ϕ.roll(+1, μ))

        dsdφ += 2 * α * ϕ

        dsdφ += 4 * λ * ϕ**3

        return dsdφ
