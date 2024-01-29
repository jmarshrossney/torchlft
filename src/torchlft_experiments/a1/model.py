from dataclasses import dataclass, asdict
from typing import TypeAlias

import torch
import torch.nn as nn
from jsonargparse.typing import PositiveInt, PositiveFloat, NonNegativeFloat

from torchlft.model import Model as BaseModel
from torchlft.scalar.actions import FreeScalarAction
from torchlft.utils.lattice import checkerboard_mask

from torchlft.transforms.core import UnivariateTransformModule
from torchlft.transforms.affine import affine_transform
from torchlft.transforms.wrappers import sum_log_gradient
from torchlft.lattice import Checkerboard2d, Lattice2d
from torchlft.nn import Activation, ConvNet2d, PointNet

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


class _CouplingLayer(nn.Module):
    def __init__(
        self,
        transform: UnivariateTransformModule,
        net: nn.Module,
        lattice: Lattice2d,
        partition_id: int,
    ):
        super().__init__()

        self.register_module("transform", transform)
        self.register_module("net", net)
        self.register_module("lattice", lattice)

        self.partition_id = partition_id

        # self.register_buffer("active_mask", None, persistent=False)
        # self.register_buffer("frozen_mask", None, persistent=False)

        # self._generate_masks_hook = self.register_forward_pre_hook(
        #    self.generate_masks
        # )

    """@staticmethod
    def generate_masks(self, inputs: Tensor):
        (φ,) = inputs
        _, *lattice, _ = φ.shape

        if self.active_mask is not None and self.active_mask.shape == lattice:
            return

        active_mask = checkerboard_mask(lattice, offset=self.offset, device=φ.device)
        frozen_mask = ~active_mask

        self.active_mask = active_mask
        self.frozen_mask = frozen_mask
    """

    def forward(self, φ: Tensor) -> tuple[Tensor, Tensor]:

        self.lattice.update_dims(*φ.shape[1:-1])
        active_mask, frozen_mask = self.lattice(self.partition_id)

        # Construct conditional transformation
        net_inputs = frozen_mask.unsqueeze(-1) * φ
        context = self.net(net_inputs)
        context = context[:, active_mask]
        transform = self.transform(context)

        φ_out = φ.clone()

        φ_out[:, active_mask], ldj = transform(φ[:, active_mask])

        return φ_out, ldj


@dataclass
class Flow:
    point_net_shape: list[int]
    point_net_activation: Activation
    spatial_net_shape: list[int]
    spatial_net_activation: Activation
    spatial_net_kernel: int

    def build(self):
        transform_cls = affine_transform()

        # Fnn applied to each point
        point_net = PointNet(
            channels=self.point_net_shape,
            activation=self.point_net_activation,
        )

        point_net = point_net.build()

        point_net.append(nn.LazyLinear(transform_cls.n_params))

        transform = UnivariateTransformModule(
            transform_cls, point_net, wrappers=[sum_log_gradient]
        )  # wrappers

        spatial_net = ConvNet2d(
            channels=self.spatial_net_shape,
            activation=self.spatial_net_activation,
            kernel_radius=self.spatial_net_kernel,
        )

        spatial_net = spatial_net.build()

        lattice = Checkerboard2d()

        return _CouplingLayer(transform, spatial_net, lattice, 0)


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
