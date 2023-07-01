from functools import partial
from typing import Optional, TypeAlias

import torch
import torch.nn.functional as F

from flows_on_spheres.nn import TransformModule
from flows_on_spheres.transforms.typing import TransformFunc, Transform

Tensor: TypeAlias = torch.Tensor


def _get_segment(
    x: Tensor, params: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    intervals, weights, omega, knots_x, knots_y = params

    i1 = torch.searchsorted(
        knots_x,
        x,
        side="right",
    )
    i0 = i1 - 1

    Δ = intervals.gather(-1, i0)
    ρ = weights.gather(-1, i1)
    ω1 = omega.gather(-1, i1)
    ω0 = omega.gather(-1, i0)
    x0 = knots_x.gather(-1, i0)
    y0 = knots_y.gather(-1, i0)

    return Δ, ρ, ω1, ω0, x0, y0


def _forward_transform(
    x: Tensor, params: tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
) -> tuple[Tensor, Tensor]:
    Δ, ρ, ω1, ω0, x0, y0 = _get_segment(x, params)

    θ = (x - x0) / Δ

    y = (
        y0
        + ρ * Δ * θ
        - ω0 * Δ**2 * θ * (1 - θ)
        + (1 / 3) * (ω1 - ω0) * Δ**2 * θ**3
    )

    dydx = ρ + ω1 * Δ * θ**2 - ω0 * Δ * (1 - θ) ** 2

    return y, dydx


def _rescale_to_interval(
    transform: TransformFunc,
    lower_bound: float,
    upper_bound: float,
    tol: float = 1e-4,
) -> TransformFunc:
    lo, up, ε = lower_bound, upper_bound, tol
    assert lo < up

    def rescaled_transform(
        x: Tensor, *args, **kwargs
    ) -> tuple[Tensor, Tensor]:
        x = torch.clamp((x - lo) / (up - lo), ε, 1 - ε)
        y, dydx = transform(x, *args, **kwargs)
        return y * (up - lo) + lo, dydx

    return rescaled_transform


def _build_spline(
    params: Tensor,
    lower_bound: float,
    upper_bound: float,
    min_interval: float,
    min_weight: float,
) -> Tensor:
    assert lower_bound < upper_bound
    assert params.isfinite().all()
    n_params = params.shape[-1]
    n_intervals = n_params // 2 - 1
    n_weights = n_params // 2 + 1
    assert n_params % 2 == 0
    intervals, weights = params.split(
        (n_intervals, n_weights),
        dim=-1,
    )
    Δ, ρ = intervals, weights

    assert n_intervals * min_interval < 1
    Δ = F.softmax(Δ, dim=-1) * (1 - min_interval * n_intervals) + min_interval

    # NOTE: this doesn't correctly enforce min_weight
    ρ = F.softplus(ρ) + min_weight
    ρ = ρ / (((ρ[..., :-2] + ρ[..., 1:-1] + ρ[..., 2:]) / 3) * Δ).sum(
        dim=-1, keepdim=True
    )

    zeros = Δ.new_zeros(*Δ.shape[:-1], 1)
    OΔO = torch.cat((zeros, Δ, zeros), dim=-1)

    ω = (ρ[..., 1:] - ρ[..., :-1]) / (OΔO[..., :-1] + OΔO[..., 1:])
    h = ρ[..., 1:-1] * Δ + (1 / 3) * (ω[..., 1:] - ω[..., :-1]) * Δ**2

    knots_x = torch.cat(
        (
            zeros,
            torch.cumsum(Δ, dim=-1),
        ),
        dim=-1,
    )
    knots_y = torch.cat(
        (
            zeros,
            torch.cumsum(h, dim=-1),
        ),
        dim=-1,
    )
    return Δ, ρ, ω, knots_x, knots_y


def bspline_transform(
    params_are_batched: bool,
    lower_bound: float,
    upper_bound: float,
    min_interval: float = 1e-2,
    min_weight: float = 1e-3,
    bounds_tol: float = 1e-4,
) -> Transform:
    build_spline = partial(
        _build_spline,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        min_interval=min_interval,
        min_weight=min_weight,
    )
    rescale_to_interval = partial(
        _rescale_to_interval,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        tol=bounds_tol,
    )

    in_dims = (0, 0) if params_are_batched else (0, None)
    vmap = partial(torch.vmap, in_dims=in_dims, out_dims=(0, 0))

    forward_fn = vmap(rescale_to_interval(_forward_transform))
    # inverse_fn = None  # TODO

    class BSplineTransform:
        def __init__(self, params: Tensor):
            self.params = build_spline(params)

        def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
            return forward_fn(x, self.params)

        def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
            raise NotImplementedError

    return BSplineTransform


class BSplineModule(TransformModule):
    def __init__(
        self,
        *,
        n_intervals: int,
        min_interval: float = 1e-2,
        min_weight: float = 1e-3,
        net_hidden_shape: Optional[list[int]] = None,
        net_activation: Optional[str] = None,
    ):
        super().__init__(
            n_params=2 * n_intervals + 2,
            net_hidden_shape=net_hidden_shape,
            net_activation=net_activation,
        )
        self._transform = bspline_transform(
            params_are_batched=net_hidden_shape is not None,
            lower_bound=-1.0,
            upper_bound=+1.0,
            min_interval=min_interval,
            min_weight=min_weight,
        )

    def transform(self, params: Tensor) -> Transform:
        return self._transform(params)
