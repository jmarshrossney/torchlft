from dataclasses import dataclass, asdict
from typing import TypeAlias

import torch
import torch.nn as nn
from jsonargparse.typing import PositiveInt, PositiveFloat, NonNegativeFloat

from torchlft.nflow.model import Model as BaseModel
from torchlft.nflow.nn import Activation, ConvNet2d, PointNet
from torchlft.nflow.partition import Checkerboard2d
from torchlft.nflow.transforms.core import UnivariateTransformModule
from torchlft.nflow.transforms.affine import affine_transform
from torchlft.nflow.transforms.wrappers import sum_log_gradient
from torchlft.nflow.utils import Composition

from torchlft.scalar.actions import FreeScalarAction

Tensor: TypeAlias = torch.Tensor


@dataclass
class Target:
    lattice: tuple[PositiveInt, PositiveInt]
    m_sq: PositiveFloat

    def __post_init__(self):
        L, T = self.lattice
        assert L % 2 == 0 and T % 2 == 0
        assert self.m_sq > 0

    def build(self) -> FreeScalarAction:
        return FreeScalarAction(m_sq=self.m_sq)


class CouplingLayer(nn.Module):
    def __init__(
        self,
        transform: UnivariateTransformModule,
        spatial_net: nn.Module,
        layer_id: int,
    ):
        super().__init__()

        self.register_module("transform", transform)
        self.register_module("spatial_net", spatial_net)

        partitioning = Checkerboard2d(partition_id=layer_id)
        self.register_module("partitioning", partitioning)

    def forward(self, φ_in: Tensor) -> tuple[Tensor, Tensor]:
        # Get active and frozen masks
        _, L, T, _ = φ_in.shape
        active_mask, frozen_mask = self.partitioning(dimensions=(L, T))

        # Construct conditional transformation
        net_inputs = frozen_mask.unsqueeze(-1) * φ_in
        context = self.spatial_net(net_inputs)[:, active_mask]
        transform = self.transform(context)

        # Transform active variables
        φ_out = φ_in.clone()
        φ_out[:, active_mask], ldj = transform(φ_in[:, active_mask])

        return φ_out, ldj


@dataclass(kw_only=True)
class AffineTransformModuleDenseNet:
    net: PointNet
    scale_fn: str = "exponential"
    symmetric: bool = False
    shift_only: bool = False
    rescale_only: bool = False

    def build(self):
        transform_cls = affine_transform()
        net = self.net.build()
        net.append(nn.LazyLinear(transform_cls.n_params))
        transform = UnivariateTransformModule(
            transform_cls, net, wrappers=[sum_log_gradient]
        )
        return transform


@dataclass
class Flow:
    transform: AffineTransformModuleDenseNet
    spatial_net: ConvNet2d
    n_blocks: int

    def build(self):
        layers = []

        for layer_id in range(2 * self.n_blocks):
            transform = self.transform.build()
            spatial_net = self.spatial_net.build()
            layer = CouplingLayer(transform, spatial_net, layer_id)

            layers.append(layer)

        return Composition(*layers)


class Model(BaseModel):
    def __init__(
        self,
        target: Target,
        flow: Flow,
    ):
        super().__init__(**asdict(target))

        self.target_action = target.build()
        self.flow = flow.build()

    def flow_forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        return self.flow(z)

    def base(self, batch_size: PositiveInt) -> tuple[Tensor, Tensor]:
        z = torch.empty(
            size=(
                batch_size,
                *self.lattice,
                1,
            ),
            device=self.device,
            dtype=self.dtype,
        ).normal_()
        S_z = (0.5 * z * z).flatten(1).sum(dim=1, keepdim=True)
        return z, S_z

    def target(self, φ: Tensor) -> Tensor:
        return self.target_action(φ)
