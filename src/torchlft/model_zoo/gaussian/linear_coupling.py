from dataclasses import dataclass, asdict
from enum import auto, StrEnum
from typing import TypeAlias

import torch
import torch.nn as nn
from jsonargparse.typing import PositiveInt

from torchlft.nflow.layer import Composition
from torchlft.nflow.nn import DenseNet
from torchlft.nflow.transforms.core import UnivariateTransformModule
from torchlft.nflow.transforms.affine import affine_transform
from torchlft.nflow.transforms.wrappers import sum_log_gradient
from torchlft.lattice.scalar.layers import DenseCouplingLayer, DiagonalLinearLayer
from torchlft.utils.lattice import checkerboard_mask

from torchlft.model_zoo.gaussian.core import GaussianModel, Target

Tensor: TypeAlias = torch.Tensor


@dataclass(kw_only=True)
class ShiftTransformModule:

    def build(self):
        return transform_module


@dataclass
class LinearCouplingFlow:
    transform: ShiftTransformModule
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
        return φ, ldj.squeeze(1)
