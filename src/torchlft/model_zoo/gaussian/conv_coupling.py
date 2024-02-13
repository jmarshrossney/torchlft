from dataclasses import dataclass, asdict, field
from typing import TypeAlias

import torch
import torch.nn as nn
from jsonargparse.typing import PositiveInt

from torchlft.nflow.nn import ConvNet2d, PointNet
from torchlft.nflow.transforms.core import UnivariateTransformModule
from torchlft.nflow.transforms.affine import affine_transform
from torchlft.nflow.transforms.wrappers import sum_log_gradient
from torchlft.nflow.utils import Composition
from torchlft.lattice.scalar.layers import CouplingLayer

from torchlft.model_zoo.gaussian.core import GaussianModel, Target

Tensor: TypeAlias = torch.Tensor


def _default_point_net():
    return PointNet(channels=[], activation="identity")


@dataclass(kw_only=True)
class AffineTransformModule:
    net: PointNet = field(default_factory=_default_point_net)
    scale_fn: str = "exponential"
    symmetric: bool = False
    shift_only: bool = False
    rescale_only: bool = False

    def build(self):
        spec = asdict(self)
        _ = spec.pop("net")
        AffineTransform = affine_transform(**spec)
        net = self.net.build()
        net.append(nn.LazyLinear(AffineTransform.n_params))
        transform_module = UnivariateTransformModule(
            transform_cls=AffineTransform,
            context_fn=net,
            wrappers=[sum_log_gradient],
        )
        return transform_module


@dataclass
class ConvCouplingFlow:
    transform: AffineTransformModule
    net: ConvNet2d
    n_blocks: PositiveInt

    def build(self):
        layers = []

        for layer_id in range(2 * self.n_blocks):
            transform_module = self.transform.build()
            net = self.net.build()
            layer = CouplingLayer(transform_module, net, layer_id)

            layers.append(layer)

        return Composition(*layers)


class Target2d(Target):
    lattice_dim = 2


class ConvCouplingModel(GaussianModel):
    def __init__(
        self,
        target: Target,
        flow: ConvCouplingFlow,
    ):
        super().__init__(target)

        self.register_module("flow", flow.build())

    def flow_forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        L = self.target.lattice_length
        z = z.view(-1, L, L, 1)
        φ, log_det_dφdz = self.flow(z)
        φ = φ.flatten(start_dim=1)
        return φ, log_det_dφdz.squeeze(1)
