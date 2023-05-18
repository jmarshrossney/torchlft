# from __future__ import annotations

import math
from functools import partial
from typing import TypeAlias, TYPE_CHECKING, Union

import torch
import torch.nn as nn

from torchlft import constraints as constraints
from torchlft.fields import CanonicalScalarField
from torchlft.networks import make_cnn
from torchlft.transforms import Translation, AffineTransform, RQSplineTransform
from torchlft.utils.lattice import make_checkerboard

if TYPE_CHECKING:
    from torchlft.typing import *

Tensor: TypeAlias = torch.Tensor

__all__ = [
    "AdditiveCouplingLayer",
    "AffineCouplingLayer",
    "RQSplineCouplingLayer",
    "NormalizingFlow",
]


class _CouplingLayer(nn.Module):
    def __init__(
        self,
        transform_n_params: int,
        **net_kwargs,
    ):
        super().__init__()
        assert n_lattice % 2 == 0, "Lattice must have an even number of sites"

        self.net_a = make_cnn(
            channels_in=1,
            channels_out=transform_n_params,
            **net_kwargs,
        )
        self.net_b = make_cnn(
            channels_in=1,
            channels_out=transform_n_params,
            **net_kwargs,
        )


class AdditiveCouplingLayer(_CouplingLayer):
    Transform = Translation

    def __init__(
        self,
        **net_kwargs,
    ):
        super().__init__(1, **net_kwargs)

    def forward(
        self, ϕ_a: Tensor, ϕ_b: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        t = self.net_b(ϕ_b.unsqueeze(1))
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


class NormalizingFlow(nn.Sequential):
    def __init__(self, *layers: CouplingLayer):
        super().__init__(*layers)

    def forward(self, ϕ: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, L, T = ϕ.shape

        mask = make_checkerboard((L, T), device=ϕ.device)
        ϕ_a = ϕ.masked_fill(mask, float("nan"))
        ϕ_b = ϕ.masked_fill(~mask, float("nan"))

        ldj_total = torch.zeros(batch_size, device=ϕ.device)

        for layer in self:
            ϕ_a, ϕ_b, ldj = layer(ϕ_a, ϕ_b)
            ldj_total += ldj

            self.on_layer(ϕ_a, ϕ_b, ldj_total)

        ϕ = ϕ_a.nan_to_num() + ϕ_b.nan_to_num()

        return ϕ, ldj_total

    def on_layer(self, ϕ_a: Tensor, ϕ_b: Tensor, ldj: Tensor) -> None:
        pass
