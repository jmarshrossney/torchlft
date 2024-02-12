from typing import TypeAlias

from jsonargparse.typing import (
    PositiveInt,
    PositiveFloat,
    restricted_number_type,
)
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchlft.nflow.model import Model as BaseModel

from torchlft.utils.lattice import laplacian
from torchlft.utils.linalg import dot, mv

Tensor: TypeAlias = torch.Tensor

LatticeDim = restricted_number_type(
    "LatticeDim", int, [("==", 1), ("==", 2)], join="or"
)

class TriangularLayer(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        D = size

        weight = torch.empty(D, D).uniform_(0, 1)
        self.register_parameter("weight", nn.Parameter(weight))

        mask = torch.ones(D, D).tril().bool()
        self.register_buffer("mask", mask)

    def get_weight(self) -> Tensor:
        return self.mask * self.weight

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        L = self.get_weight()
        x = mv(L, z)
        log_det_dxdz = L.diag().log().sum().expand(x.shape[0])
        return x, log_det_dxdz


class Model(BaseModel):
    def __init__(
        self,
        lattice_length: PositiveInt,
        lattice_dim: LatticeDim,
        m_sq: PositiveFloat,
    ):
        super().__init__()
        L, d = lattice_length, lattice_dim
        D = pow(L, d)

        self.lattice_size = D
        self.m_sq = m_sq

        K = -laplacian(L, d) + m_sq * torch.eye(D)
        Σ = torch.linalg.inv(K)
        T = torch.linalg.cholesky(Σ)
        self.kernel = K
        self.covariance = Σ
        self.cholesky = T

        self.register_module("linear", TriangularLayer(D))

    def flow_forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        return self.linear(z)

    def base(self, batch_size: PositiveInt) -> tuple[Tensor, Tensor]:
        z = torch.empty(
            size=(batch_size, self.lattice_size),
            device=self.device,
            dtype=self.dtype,
        ).normal_()
        S_z = 0.5 * dot(z, z)
        return z, S_z

    def target(self, φ: Tensor) -> Tensor:
        K = self.kernel
        return 0.5 * dot(φ, mv(K, φ))
