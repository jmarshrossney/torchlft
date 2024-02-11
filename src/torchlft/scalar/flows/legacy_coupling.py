from dataclasses import dataclass, asdict
from typing import TypeAlias

import torch
import torch.nn as nn
from jsonargparse.typing import PositiveInt, PositiveFloat

from torchlft.nflow.model import Model as BaseModel
from torchlft.nflow.nn import Activation, ConvNet2d, PointNet
from torchlft.nflow.partition import Checkerboard2d
from torchlft.nflow.transforms.core import UnivariateTransformModule
from torchlft.nflow.transforms.affine import affine_transform
from torchlft.nflow.transforms.wrappers import sum_log_gradient
from torchlft.nflow.utils import Composition

from torchlft.scalar.action import ActionV1
from torchlft.scalar.coupling_layers import LegacyCouplingLayer

Tensor: TypeAlias = torch.Tensor

@dataclass
class Target:
    lattice_length: PositiveInt
    lattice_dim: PositiveInt
    m_sq: PositiveFloat

    def __post_init__(self):
        assert self.lattice_length % 2 == 0
        assert self.lattice_dim in (1, 2)
        assert self.m_sq > 0

    def build(self) -> ActionV1:
        return ActionV1(**asdict(self))


@dataclass(kw_only=True)
class AffineTransformModule:
    scale_fn: str = "exponential"
    symmetric: bool = False
    shift_only: bool = False
    rescale_only: bool = False

    def build(self):
        transform_cls = affine_transform()
        net = nn.Identity()
        #nn.LazyLinear((lattice_size // 2) * transform_cls.n_params)
        transform = UnivariateTransformModule(
            transform_cls, net, wrappers=[sum_log_gradient]
        )
        return transform


@dataclass
class Flow:
    transform: AffineTransformModule
    net: PointNet
    n_blocks: int

    def build(self, lattice_size: int):
        layers = []

        for layer_id in range(2 * self.n_blocks):
            transform = self.transform.build()
            size_out = (lattice_size // 2) * transform.transform_cls.n_params
            net = self.net.build()
            net.append(nn.LazyLinear(size_out))
            layer = LegacyCouplingLayer(transform, net, layer_id)
            layers.append(layer)

        return Composition(*layers)


class Model(BaseModel):
    def __init__(
        self,
        target: Target,
        flow: Flow,
    ):
        super().__init__()

        self.lattice_size = pow(target.lattice_length, target.lattice_dim)
        self.target_action = target.build()
        self.register_module("flow", flow.build(self.lattice_size))

    def flow_forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        # permute to checkerboard
        φ, ldj = self.flow(z)
        # permute back
        return φ, ldj

    def base(self, batch_size: PositiveInt) -> tuple[Tensor, Tensor]:
        z = torch.empty(
            size=(
                batch_size,
                self.lattice_size,
            ),
            device=self.device,
            dtype=self.dtype,
        ).normal_()
        S_z = (0.5 * z * z).sum(dim=1, keepdim=True)
        return z, S_z

    def target(self, φ: Tensor) -> Tensor:
        return self.target_action(φ)
