from dataclasses import dataclass, asdict
from enum import auto, StrEnum
from math import log, pi as π
from typing import TypeAlias

from jsonargparse.typing import (
    PositiveInt,
    PositiveFloat,
    restricted_number_type,
)
import torch

from torchlft.nflow.model import Model as BaseModel
from torchlft.nflow.layer import Composition
from torchlft.nflow.nn import DenseNet
from torchlft.nflow.transforms.core import UnivariateTransformModule
from torchlft.nflow.transforms.affine import affine_transform
from torchlft.nflow.transforms.wrappers import sum_log_gradient
from torchlft.lattice.scalar.action import GaussianAction
from torchlft.lattice.scalar.layers import (
    DiagonalLinearLayer,
    TriangularLinearLayer,
    DenseCouplingLayer,
    CouplingLayer,
)
from torchlft.utils.lattice import checkerboard_mask
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


class ValidPartitioning(StrEnum):
    lexicographic = auto()
    checkerboard = auto()
    random = auto()

    def build(self, lattice_length: int, lattice_dim: int):
        L, d = lattice_length, lattice_dim
        D = pow(L, d)

        if str(self) == "lexicographic":
            return torch.arange(D)

        elif str(self) == "checkerboard":
            checker = checkerboard_mask([L for _ in range(d)]).flatten()
            output_indices = torch.cat(
                [torch.argwhere(checker), torch.argwhere(~checker)]
            ).squeeze(1)
            _, input_indices = output_indices.sort()
            return input_indices

        elif str(self) == "random":
            return torch.randperm(D)


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

        return z, (S + log_norm).unsqueeze(-1)

    def compute_target(self, φ: Tensor) -> Tensor:
        return self.target(φ).unsqueeze(-1)


class TriangularLinearModel(GaussianModel):
    def __init__(self, target: Target):
        super().__init__(target)

        D = self.target.lattice_size
        self.register_module("transform", TriangularLinearLayer(D))

    def flow_forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        return self.transform(z)


@dataclass
class LinearCouplingFlow:
    n_layers: PositiveInt

    def build(self, lattice_size: int):
        layers = []

        for layer_id in range(self.n_layers):
            transform_module = UnivariateTransformModule(
                transform_cls=affine_transform(shift_only=True),
                context_fn=nn.Identity(),
                wrappers=[sum_log_gradient],
            )

            linear = nn.LazyLinear(lattice_size // 2, bias=False)

            layer = DenseCouplingLayer(transform_module, linear, layer_id)

            layers.append(layer)

        layers.append(DiagonalLinearLayer(lattice_size))

        return Composition(*layers)


class LinearCouplingModel(GaussianModel):
    def __init__(
        self,
        target: Target,
        flow: LinearCouplingFlow,
        partitioning: ValidPartitioning,
    ):
        super().__init__(target)

        self.register_module("flow", flow.build(self.target.lattice_size))

        partitioning = ValidPartitioning(str(partitioning))
        indices = partitioning.build(
            self.target.lattice_length, self.target.lattice_dim
        )
        self.register_buffer("indices", indices)

    def flow_forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        φ, ldj = self.flow(z)
        φ = φ[:, self.indices]  # interleave elements to undo partitioning
        return φ, ldj


@dataclass
class NonLinearCouplingFlow:
    net: DenseNet
    n_layers: PositiveInt
    shift_only: bool = False

    def build(self, lattice_size: int):
        layers = []

        for layer_id in range(self.n_layers):
            transform_module = UnivariateTransformModule(
                transform_cls=affine_transform(shift_only=self.shift_only),
                context_fn=nn.Identity(),
                wrappers=[sum_log_gradient],
            )
            net = self.net.build()
            size_out = (lattice_size // 2) if self.shift_only else lattice_size
            net.append(nn.LazyLinear(size_out))
            layer = DenseCouplingLayer(transform_module, net, layer_id)
            layers.append(layer)

        if self.shift_only:
            layers.append(DiagonalLinearLayer(lattice_size))

        return Composition(*layers)


class NonLinearCouplingModel(LinearCouplingModel):
    def __init__(
        self,
        target: Target,
        flow: NonLinearCouplingFlow,
        partitioning: ValidPartitioning,
    ):
        super().__init__(target, flow, partitioning)


@dataclass
class EquivLinearCouplingFlow:
    n_layers: int
    radius: int

    def build(self, lattice_size: int):
        layers = []

        for layer_id in range(self.n_layers):
            transform_module = UnivariateTransformModule(
                transform_cls=affine_transform(shift_only=True),
                context_fn=nn.LazyLinear(1, bias=False),
                wrappers=[sum_log_gradient],
            )

            layer = CouplingLayer(transform_module, self.radius, layer_id)

            layers.append(layer)

        layers.append(DiagonalLinearLayer(lattice_size))

        return Composition(*layers)


class EquivLinearCouplingModel(GaussianModel):
    def __init__(
        self,
        target: Target,
        flow: EquivLinearCouplingFlow,
    ):
        assert target.lattice_dim == 2
        super().__init__(target)

        self.register_module("flow", flow.build())

    def flow_forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        L = self.target.lattice_length
        z = z.view(-1, L, L, 1)
        φ, log_det_dφdz = self.flow(z)
        φ = φ.flatten(start_dim=1)
        return φ, log_det_dφdz
