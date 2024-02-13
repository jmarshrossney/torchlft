from dataclasses import dataclass, asdict
from math import log, pi as π
from typing import TypeAlias

from jsonargparse.typing import (
    PositiveInt,
    PositiveFloat,
    restricted_number_type,
)
import torch

from torchlft.lattice.scalar.action import GaussianAction
from torchlft.lattice.scalar.layers import TriangularLinearLayer
from torchlft.nflow.model import Model as BaseModel

from torchlft.utils.linalg import dot

Tensor: TypeAlias = torch.Tensor

LatticeDim = restricted_number_type(
    "LatticeDim", int, [("==", 1), ("==", 2)], join="or"
)


@dataclass
class Target:
    lattice_length: PositiveInt
    lattice_dim: LatticeDim
    m_sq: PositiveFloat

    def __post_init__(self):
        assert self.lattice_length % 2 == 0

    def build(self) -> GaussianAction:
        return GaussianAction(**asdict(self))


class GaussianModel(BaseModel):
    def __init__(
        self,
        target: Target,
    ):
        super().__init__()

        self.register_module("target", target.build())

    def sample_base(self, batch_size: PositiveInt) -> tuple[Tensor, Tensor]:
        D = self.target.lattice_size

        z = torch.empty(
            size=(batch_size, D), device=self.device, dtype=self.dtype
        ).normal_()

        S = 0.5 * dot(z, z)
        log_norm = 0.5 * D * log(2 * π)

        return z, S + log_norm

    def compute_target(self, φ: Tensor) -> Tensor:
        return self.target(φ)


class TriangularLinearModel(GaussianModel):
    def __init__(self, target: Target):
        super().__init__(target)

        D = self.target.lattice_size
        self.register_module("transform", TriangularLinearLayer(D))

    def flow_forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        return self.transform(z)
