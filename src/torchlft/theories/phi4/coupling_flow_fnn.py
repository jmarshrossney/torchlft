# from __future__ import annotations

import math
from functools import partial
from typing import TypeAlias, TYPE_CHECKING, Union

import torch
import torch.nn as nn

from torchlft import constraints as constraints
from torchlft.fields import CanonicalScalarField
from torchlft.networks import make_fnn
from torchlft.transforms import Translation, AffineTransform, RQSplineTransform
from torchlft.utils.lattice import make_checkerboard

if TYPE_CHECKING:
    from torchlft.typing import *

Tensor: TypeAlias = torch.Tensor

__all__ = [
    "AdditiveCouplingLayer",
    "AffineCouplingLayer",
    "RQSplineCouplingLayer",
    "Geometry",
    "NormalizingFlow",
]


class _CouplingLayer(nn.Module):
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


class AdditiveCouplingLayer(_CouplingLayer):
    Transform = Translation

    def __init__(
        self,
        lattice_shape: torch.Size,
        **net_kwargs,
    ):
        super().__init__(lattice_shape, 1, **net_kwargs)

    def forward(
        self, ϕ_a: Tensor, ϕ_b: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        t = self.net_b(ϕ_b)
        f = Translation(t)
        ϕ_a, ldj_a = f(ϕ_a)

        t = self.net_b(ϕ_a)
        f = Translation(t)
        ϕ_b, ldj_b = f(ϕ_b)

        return ϕ_a, ϕ_b, ldj_a + ldj_b


class AffineCouplingLayer(_CouplingLayer):
    def __init__(
        self,
        lattice_shape: torch.Size,
        **net_kwargs,
    ):
        super().__init__(lattice_shape, 2, **net_kwargs)

    def forward(
        self, ϕ_a: Tensor, ϕ_b: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        s, t = self.net_b(ϕ_b).tensor_split(2, dim=1)
        f = AffineTransform(s, t)
        ϕ_a, ldj_a = f(ϕ_a)

        s, t = self.net_b(ϕ_a).tensor_split(2, dim=1)
        f = AffineTransform(s, t)
        ϕ_b, ldj_b = f(ϕ_b)

        return ϕ_a, ϕ_b, ldj_a + ldj_b


class RQSplineCouplingLayer(_CouplingLayer):
    def __init__(
        self,
        lattice_shape: list[int],
        *,
        n_segments: int,
        upper_bound: float,
        **net_kwargs,
    ):
        super().__init__(lattice_shape, (3 * n_segments - 1), **net_kwargs)

        self.n_segments = n_segments
        self.upper_bound = float(upper_bound)

    def forward(
        self, ϕ_a: Tensor, ϕ_b: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        n = self.n_segments
        w, h, d = (
            self.net_b(ϕ_b)
            .unflatten(1, (-1, 3 * n - 1))
            .split([n, n, n - 1], dim=2)
        )
        f = RQSplineTransform(
            w,
            h,
            d,
            lower_bound=-self.upper_bound,
            upper_bound=self.upper_bound,
            bounded=False,
            periodic=False,
            min_slope=1e-3,
        )
        ϕ_a, ldj_a = f(ϕ_a)

        w, h, d = (
            self.net_a(ϕ_a)
            .unflatten(1, (-1, 3 * n - 1))
            .split([n, n, n - 1], dim=2)
        )
        f = RQSplineTransform(
            w,
            h,
            d,
            lower_bound=-self.upper_bound,
            upper_bound=self.upper_bound,
            bounded=False,
            periodic=False,
            min_slope=1e-3,
        )
        ϕ_b, ldj_b = f(ϕ_b)

        return ϕ_a, ϕ_b, ldj_a + ldj_b


CouplingLayer = Union[
    AdditiveCouplingLayer, AffineCouplingLayer, RQSplineCouplingLayer
]

class Geometry(nn.Module):

    def __init__(self, lattice_shape: tuple[int]):
        super().__init__()
        self.lattice_shape = lattice_shape
        self.register_buffer("mask", make_checkerboard(lattice_shape, device="cpu"))

    def partition(self, ϕ: Tensor) -> tuple[Tensor, Tensor]:
        return ϕ[:, self.mask], ϕ[:, ~self.mask]

    def restore(self, ϕ_a: Tensor, ϕ_b: Tensor) -> Tensor:
        shape = [ϕ_a.shape[0]] + list(self.mask.shape)
        ϕ = torch.empty(shape, device=ϕ_a.device, dtype=ϕ_a.dtype)
        ϕ[:, self.mask] = ϕ_a
        ϕ[:, ~self.mask] = ϕ_b
        return ϕ


class NormalizingFlow(nn.Module):
    def __init__(self, geometry: Geometry, *layers: CouplingLayer):
        super().__init__()
        self.geometry = geometry

        self.layers = nn.ModuleList(layers)

        for layer in self.layers:
            layer.geometry = geometry

    def forward(self, ϕ: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, L, T = ϕ.shape

        #mask = make_checkerboard((L, T), device=ϕ.device)
        #ϕ_a, ϕ_b = ϕ[:, mask], ϕ[:, ~mask]
        ϕ_a, ϕ_b = self.geometry.partition(ϕ)

        ldj_total = torch.zeros(batch_size, device=ϕ.device)

        for layer in self.layers:
            ϕ_a, ϕ_b, ldj = layer(ϕ_a, ϕ_b)
            ldj_total += ldj

            self.on_layer(ϕ_a, ϕ_b, ldj_total)

        ϕ = self.geometry.restore(ϕ_a, ϕ_b)

        #ϕ = torch.empty((batch_size, L, T), device=ϕ_a.device, dtype=ϕ_a.dtype)
        #ϕ[:, mask] = ϕ_a
        #ϕ[:, ~mask] = ϕ_b

        return ϕ, ldj_total

    def on_layer(self, ϕ_a: Tensor, ϕ_b: Tensor, ldj: Tensor) -> None:
        pass
