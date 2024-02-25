from enum import StrEnum, auto
from typing import TypeAlias
from dataclasses import dataclass

from jsonargparse.typing import PositiveInt, PositiveFloat, NonNegativeFloat
import torch
import torch.nn as nn

from torchlft.nflow.model import Model as BaseModel
from torchlft.nflow.layer import Composition
from torchlft.nflow.nn import DenseNet
from torchlft.nflow.transforms.core import UnivariateTransformModule
from torchlft.nflow.transforms.affine import affine_transform
from torchlft.nflow.transforms.spline import spline_transform
from torchlft.nflow.transforms.wrappers import sum_log_gradient
from torchlft.lattice.scalar.action import Phi4Action
from torchlft.lattice.scalar.layers import (
    GlobalRescalingLayer,
    DenseCouplingLayer,
    CouplingLayer,
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
class Target:
    lattice_length: PositiveInt
    β: PositiveFloat
    λ: NonNegativeFloat

    def __post_init__(self):
        assert self.lattice_length % 2 == 0

    def build(self) -> Phi4Action:
        return Phi4Action(β=self.β, λ=self.λ)
        # return Phi4Action(m_sq=self.m_sq, λ=self.λ)


@dataclass
class AffineCouplingFlow:
    net: DenseNet
    n_layers: PositiveInt
    global_rescale: bool
    symmetric: bool

    def build(self, lattice_size: int):
        layers = []

        for layer_id in range(self.n_layers):
            transform_module = UnivariateTransformModule(
                transform_cls=affine_transform(symmetric=self.symmetric),
                context_fn=nn.Identity(),  # nn.Tanh(),
                wrappers=[sum_log_gradient],
            )
            net = self.net.build()
            net.append(nn.LazyLinear(lattice_size, bias=self.net.bias))
            layer = DenseCouplingLayer(transform_module, net, layer_id)
            layers.append(layer)

        if self.global_rescale:
            layers.append(GlobalRescalingLayer())

        return Composition(*layers)


@dataclass
class SplineCouplingFlow:
    net: DenseNet
    n_layers: PositiveInt
    global_rescale: bool
    n_spline_bins: int

    def build(self, lattice_size: int):
        layers = []

        for layer_id in range(self.n_layers):
            transform_module = UnivariateTransformModule(
                transform_cls=spline_transform(
                    n_bins=self.n_spline_bins,
                    lower_bound=-5.0,
                    upper_bound=+5.0,
                    boundary_conditions="linear",
                ),
                context_fn=nn.Identity(),
                wrappers=[sum_log_gradient],
            )
            net = self.net.build()
            size_out = transform_module.transform_cls.n_params * (
                lattice_size // 2
            )
            net.append(nn.LazyLinear(size_out))
            layer = DenseCouplingLayer(transform_module, net, layer_id)
            layers.append(layer)

        if self.global_rescale:
            layers.append(GlobalRescalingLayer())

        return Composition(*layers)


class DenseCouplingModel(BaseModel):
    def __init__(
        self,
        target: Target,
        flow: SplineCouplingFlow,
        partitioning: ValidPartitioning,
    ):
        super().__init__()

        L = target.lattice_length  # TODO
        self.L = L

        self.register_module("target", target.build())

        self.register_module("flow", flow.build(L**2))

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
        φ = φ.unflatten(1, (self.L, self.L, 1))
        return self.target(φ)

    def flow_forward(self, z: Tensor) -> Tensor:
        φ, ldj = self.flow(z)
        φ = φ[:, self.indices]  # interleave elements to undo partitioning
        return φ, ldj

    def grad_pullback(self, z: Tensor) -> Tensor:
        z.requires_grad_(True)
        z.grad = None

        φ, ldj = self.flow_forward(z)
        S = self.compute_target(φ) - ldj

        (gradient,) = torch.autograd.grad(
            outputs=S,
            inputs=z,
            grad_outputs=torch.ones_like(S),
        )

        z.requires_grad_(False)
        z.grad = None

        return gradient
