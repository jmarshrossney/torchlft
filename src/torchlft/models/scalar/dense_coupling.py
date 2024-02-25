from enum import StrEnum, auto
from typing import TypeAlias
from dataclasses import dataclass

from jsonargparse.typing import (
    PositiveInt,
    PositiveFloat,
    NonNegativeFloat
)
import torch
import torch.nn as nn

from torchlft.nflow.model import Model as BaseModel
from torchlft.nflow.layer import Composition
from torchlft.nflow.nn import DenseNet
from torchlft.nflow.transforms.core import UnivariateTransformModule
from torchlft.nflow.transforms.affine import affine_transform
from torchlft.nflow.transforms.spline import spline_transform
from torchlft.nflow.transforms.wrappers import sum_log_gradient
from torchlft.nflow.utils import compute_grad_pullback
from torchlft.utils.lattice import laplacian
from torchlft.lattice.scalar.action import Phi4Action
from torchlft.lattice.scalar.layers import (
    GlobalRescalingLayer,
    TriangularLinearLayer,
    DenseCouplingLayer,
)
from torchlft.utils.lattice import checkerboard_mask
from torchlft.utils.linalg import dot

Tensor: TypeAlias = torch.Tensor


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


@dataclass
class Phi4Target:
    lattice_length: PositiveInt
    β: PositiveFloat
    λ: NonNegativeFloat

    def __post_init__(self):
        assert self.lattice_length % 2 == 0

    def build(self) -> Phi4Action:
        lattice = (self.lattice_length, self.lattice_length)
        return Phi4Action(lattice, β=self.β, λ=self.λ)
        # return Phi4Action(m_sq=self.m_sq, λ=self.λ)


@dataclass
class AffineTransform:
    symmetric: bool

    def build(self):
        return affine_transform(symmetric=self.symmetric)


@dataclass
class SplineTransform:
    n_bins: PositiveInt
    bounds: PositiveFloat = 5.0

    def build(self):
        return spline_transform(
            n_bins=self.n_bins,
            lower_bound=-self.bounds,
            upper_bound=+self.bounds,
            boundary_conditions="linear",
        )


@dataclass
class FreeTheoryLayer:
    m_sq: PositiveFloat | None = None
    frozen: bool = True

    def build(self, target: Phi4Target):
        L = target.lattice_length
        kernel = -laplacian(L, 2) + self.m_sq * torch.eye(L**2)
        layer = TriangularLinearLayer.from_gaussian_target(precision=kernel)
        layer.requires_grad_(not self.frozen)
        return layer


@dataclass
class AffineCouplingBlock:
    transform: AffineTransform
    net: DenseNet
    n_layers: PositiveInt

    def build(self, target: Phi4Target):
        layers = []

        for layer_id in range(self.n_layers):

            transform_module = UnivariateTransformModule(
                transform_cls=self.transform.build(),
                context_fn=nn.Identity(),
                wrappers=[sum_log_gradient],
            )

            net = self.net.build()
            size_out = transform_module.transform_cls.n_params * (
                target.lattice_length**2 // 2
            )
            net.append(nn.LazyLinear(size_out))

            layer = DenseCouplingLayer(transform_module, net, layer_id)
            layers.append(layer)

        return Composition(*layers)


@dataclass
class SplineCouplingBlock:
    transform: SplineTransform
    net: DenseNet
    n_layers: PositiveInt

    def build(self, target: Phi4Target):
        layers = []

        for layer_id in range(self.n_layers):

            transform_module = UnivariateTransformModule(
                transform_cls=self.transform.build(),
                context_fn=nn.Identity(),
                wrappers=[sum_log_gradient],
            )

            net = self.net.build()
            size_out = transform_module.transform_cls.n_params * (
                target.lattice_length**2 // 2
            )
            net.append(nn.LazyLinear(size_out))

            layer = DenseCouplingLayer(transform_module, net, layer_id)
            layers.append(layer)

        return Composition(*layers)


@dataclass
class GlobalRescaling:
    init_scale: PositiveFloat = 1
    frozen: bool = False

    def build(self, target: Phi4Target):
        layer = GlobalRescalingLayer(self.init_scale)
        layer.requires_grad_(not self.frozen)
        return layer


@dataclass
class AffineSplineFlow:
    to_free: FreeTheoryLayer | None = None
    affine: AffineCouplingBlock | None = None
    spline: SplineCouplingBlock | None = None
    rescale: GlobalRescaling | None = None

    def build(self, target: Phi4Target):
        blocks = []
        for block in (self.to_free, self.affine, self.spline, self.rescale):
            if block is not None:
                blocks.append(block.build(target))
        assert len(blocks) >= 1
        return Composition(*blocks)


class DenseCouplingModel(BaseModel):
    def __init__(
        self,
        flow: AffineSplineFlow,
        target: Phi4Target,
        partitioning: ValidPartitioning,
    ):
        super().__init__()

        L = target.lattice_length  # TODO
        self.L = L

        self.register_module("target", target.build())

        self.register_module("flow", flow.build(target))

        partitioning = ValidPartitioning(str(partitioning))
        indices = partitioning.build(L, 2)
        self.register_buffer("indices", indices)

    def sample_base(self, batch_size: PositiveInt) -> tuple[Tensor, Tensor]:
        D = self.L**2  # target.lattice_size

        z = torch.empty(
            size=(batch_size, D), device=self.device, dtype=self.dtype
        ).normal_()

        S = 0.5 * dot(z, z)

        return z, S.unsqueeze(-1)

    def compute_target(self, φ: Tensor) -> Tensor:
        φ = φ.unflatten(1, (*self.target.lattice, 1))
        return self.target(φ)

    def flow_forward(self, z: Tensor) -> Tensor:
        φ, ldj = self.flow(z)
        φ = φ[:, self.indices]  # interleave elements to undo partitioning
        return φ, ldj

    def grad_pullback(self, z: Tensor) -> Tensor:
        return compute_grad_pullback(self, z)
