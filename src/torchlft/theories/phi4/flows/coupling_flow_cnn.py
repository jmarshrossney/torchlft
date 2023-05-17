from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from torchlft.fields import PartitionedField
from torchlft.networks import make_fnn
from torchlft.transforms import Translation, AffineTransform, RQSplineTransform
from torchlft.utils.lattice import make_checkerboard_partitions

if TYPE_CHECKING:
    from torchlft.typing import *


class MaskedScalarField(MaskedField):
    get_masks = make_checkerboard_partitions


class _CouplingLayerFNN(nn.Module):
    def __init__(
        self,
        lattice_shape: torch.Size,
        transform_n_params: int,
        **net_kwargs,
    ):
        super().__init__()
        n_lattice = math.prod(lattice_shape)
        assert n_lattice % 2 == 0, "Lattice must have an even number of sites"
        half_lattice = n_lattice // 2

        self.net_a = make_fnn(
            size_in=half_lattice,
            size_out=half_lattice * transform_n_params,
            **net_kwargs,
        )
        self.net_b = make_fnn(
            size_in=half_lattice,
            size_out=half_lattice * transform_n_params,
            **net_kwargs,
        )


class AdditiveCouplingLayerFNN(_CouplingLayerFNN):
    def __init__(
        self,
        lattice_shape: torch.Size,
        **net_kwargs,
    ):
        super().__init__(lattice_shape, 1, **net_kwargs)

    def forward(self, Φ: PartitionedScalarField) -> PartitionedScalarField:
        ϕ_a, ϕ_b = Φ.data

        t = self.net_b(ϕ_b)
        f = Translation(t)
        ϕ_a, ldj_a = f(ϕ_a)

        t = self.net_b(ϕ_a)
        f = Translation(t)
        ϕ_b, ldj_b = f(ϕ_b)

        return


class AffineCouplingLayerFNN(_CouplingLayerFNN):
    def __init__(
        self,
        lattice_shape: torch.Size,
        **net_kwargs,
    ):
        super().__init__(lattice_shape, 2, **net_kwargs)

    def forward(self, Φ: PartitionedScalarField) -> PartitionedScalarField:
        ϕ_a, ϕ_b = Φ.data

        s, t = self.net_b(ϕ_b).tensor_split(2, dim=1)
        f = AffineTransform(s, t)
        ϕ_a, ldj_a = f(ϕ_a)

        s, t = self.net_b(ϕ_a).tensor_split(2, dim=1)
        f = AffineTransform(s, t)
        ϕ_b, ldj_b = f(ϕ_b)

        return


class RQSplineCouplingLayerFNN(_CouplingLayerFNN):
    def __init__(
        self,
        lattice_shape: list[int],
        *,
        n_segments: int,
        upper_bound: float,
        **net_kwargs,
    ):
        super().__init__(lattice_shape, (2 * n_segments - 1), **net_kwargs)

        self.SplineTransform = partial(
            RQSplineTransform,
            lower_bound=-upper_bound,
            upper_bound=upper_bound,
            bounded=False,
            periodic=False,
        )

    def forward(self, Φ: PartitionedScalarField) -> PartitionedScalarField:
        ϕ_a, ϕ_b = Φ.data

        w, h, d = self.net_b(ϕ_b).tensor_split(2, dim=1)
        f = self.SplineTransform(w, h, d)
        ϕ_a, ldj_a = f(ϕ_a)

        w, h, d = self.net_b(ϕ_a).tensor_split(2, dim=1)
        f = self.SplineTransform(w, h, d)
        ϕ_b, ldj_b = f(ϕ_b)


# class CouplingFlowFNN(NormalizingFlow):
#    field_class: PartitionedScalarField


class NormalizingFlow(torch.nn.Sequential):
    def forward(self, Φ: CanonicalScalarField) -> CanonicalScalarField:

        self.input_hook(Φ)

        Φ = self.field_class.from_canonical(Φ)

        if "log_det_jacobian" not in Φ.metadata.keys():
            Φ.metadata["log_det_jacobian"] = torch.zeros(
                Φ.batch_size, device=Φ.device
            )

        for layer in self:
            Φ, log_det_jacobian = layer(Φ)

            self.layer_hook(Φ, log_det_jacobian)

            Φ.metadata["log_det_jacobian"] += log_det_jacobian

        Φ = Φ.to_canonical()

        self.output_hook(Φ)

        return Φ

    def input_hook(self):
        ...

    def layer_hook(self):
        ...

    def output_hook(self):
        ...
