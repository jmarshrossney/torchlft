"""
References:
    Gregory, J. A. & Delbourgo, R. C2 Rational \
    Quadratic Spline Interpolation to Monotonic Data, IMA Journal of \
    Numerical Analysis, 1983, 3, 141-152
    """
from functools import partial
from math import log, pi as π
from typing import Optional, TypeAlias
import warnings

import torch
import torch.nn.functional as F

from flows_on_spheres.nn import TransformModule, CircularTransformModule
from flows_on_spheres.transforms.typing import TransformFunc, Transform

Tensor: TypeAlias = torch.Tensor

warnings.filterwarnings("ignore", message="torch.searchsorted")


def _get_segment(
    input: Tensor, knots: Tensor, inverse: bool = False
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    assert input.shape == (1,)
    assert knots.dim() == 1
    assert len(knots) % 3 == 0
    knots_x, knots_y, knots_dydx = knots.tensor_split(3)

    i1 = torch.searchsorted(knots_y if inverse else knots_x, input)

    i0_i1 = torch.cat((i1 - 1, i1)).expand(3, -1)
    knots = torch.stack((knots_x, knots_y, knots_dydx))

    (x0, x1), (y0, y1), (d0, d1) = knots.gather(-1, i0_i1)

    return x0, x1, y0, y1, d0, d1


def _forward_transform(
    x: Tensor,
    knots: Tensor,
) -> tuple[Tensor, Tensor]:
    x0, x1, y0, y1, d0, d1 = _get_segment(x, knots)
    s = (y1 - y0) / (x1 - x0)

    θx = (x - x0) / (x1 - x0)

    denom = s + (d1 + d0 - 2 * s) * θx * (1 - θx)

    θy = (s * θx**2 + d0 * θx * (1 - θx)) / denom

    y = y0 + (y1 - y0) * θy

    dydx = (
        s**2
        * (d1 * θx**2 + 2 * s * θx * (1 - θx) + d0 * (1 - θx) ** 2)
        / denom**2
    )
    return y, dydx


def _inverse_transform(y: Tensor, knots: Tensor) -> tuple[Tensor, Tensor]:
    x0, x1, y0, y1, d0, d1 = _get_segment(y, knots, inverse=True)
    s = (y1 - y0) / (x1 - x0)

    θy = (y - y0) / (y1 - y0)

    b = d0 - (d1 + d0 - 2 * s) * θy
    a = s - b
    c = -s * θy

    θx = (-2 * c) / (b + (b**2 - 4 * a * c).sqrt())

    x = x0 + (x1 - x0) * θx

    denom = s + (d1 + d0 - 2 * s) * θx * (1 - θx)

    dydx = (
        s**2
        * (d1 * θx**2 + 2 * s * θx * (1 - θx) + d0 * (1 - θx) ** 2)
        / denom**2
    )
    dxdy = 1 / dydx

    return x, dxdy


# NOTE: this masks errors where the inputs fall outside the bounds,
# but I can't add a check here since vmapping would break
# Make sure to check elsewhere that inputs are within bounds
def _clamp_to_interval(
    transform: TransformFunc,
    lower_bound: float,
    upper_bound: float,
    tol: float,
) -> TransformFunc:
    assert lower_bound < upper_bound

    def _transform(x: Tensor, knots: Tensor):
        y, dydx = transform(
            x.clamp(lower_bound + tol, upper_bound - tol), knots
        )
        return y, dydx

    return _transform


def _extend_to_reals(
    transform: TransformFunc,
    lower_bound: float,
    upper_bound: float,
    tol: float,
) -> TransformFunc:
    assert lower_bound < upper_bound

    def _transform(x: Tensor, knots: Tensor):
        inside_bounds = (x > lower_bound + tol) | (x < upper_bound - tol)

        y, dydx = transform(
            x.clamp(lower_bound + tol, upper_bound - tol), knots
        )

        y = torch.where(inside_bounds, y, x)
        dydx = torch.where(inside_bounds, dydx, torch.ones_like(dydx))

        return y, dydx

    return _transform


def _build_spline(
    params: Tensor,
    *,
    lower_bound: float,
    upper_bound: float,
    periodic: bool,
    linear_tails: bool,
    min_width: float = 1e-3,
    min_height: float = 1e-3,
    min_deriv: float = 1e-3,
) -> Tensor:
    assert lower_bound < upper_bound
    assert not (periodic and linear_tails)
    assert params.isfinite().all()
    n_params = params.shape[-1]
    if periodic:
        assert n_params % 3 == 0
        n_bins = n_params // 3
        n_derivs = n_bins
    elif linear_tails:
        assert (n_params + 1) % 3 == 0
        n_bins = (n_params + 1) // 3
        n_derivs = n_bins - 1
    else:
        assert (n_params - 1) % 3 == 0
        n_bins = (n_params - 1) // 3
        n_derivs = n_bins + 1
    assert min_width * n_bins < 1
    assert min_height * n_bins < 1

    widths, heights, derivs = params.split((n_bins, n_bins, n_derivs), dim=-1)

    # Normalise the widths and heights to the interval
    widths = (
        F.softmax(widths, dim=-1) * (1 - min_width * n_bins) + min_width
    ) * (upper_bound - lower_bound)
    heights = (
        F.softmax(heights, dim=-1) * (1 - min_height * n_bins) + min_height
    ) * (upper_bound - lower_bound)

    # Ensure the derivatives are positive and > min_slope
    # Specifying β = log(2) / (1 - ε) means softplus(0, β) = 1
    derivs = F.softplus(derivs, beta=log(2) / (1 - min_deriv)) + min_deriv

    if periodic:
        derivs = torch.cat((derivs, derivs[..., 0:1]), dim=-1)
    elif linear_tails:
        ones = derivs.new_ones(*derivs.shape[:-1], 1)
        derivs = torch.cat((ones, derivs, ones), dim=-1)

    # Build the spline
    zeros = widths.new_zeros((*widths.shape[:-1], 1))
    knots_x = torch.cat(
        (
            zeros,
            torch.cumsum(widths, dim=-1),
        ),
        dim=-1,
    ).add(lower_bound)
    knots_y = torch.cat(
        (
            zeros,
            torch.cumsum(heights, dim=-1),
        ),
        dim=-1,
    ).add(lower_bound)
    knots_dydx = derivs

    knots = torch.cat((knots_x, knots_y, knots_dydx), dim=-1)

    return knots


def rqspline_transform(
    params_are_batched: bool,
    lower_bound: float,
    upper_bound: float,
    periodic: bool = False,
    linear_tails: bool = False,
    min_width: float = 1e-3,
    min_height: float = 1e-3,
    min_deriv: float = 1e-3,
    bounds_tol: float = 1e-4,
) -> Transform:
    build_spline = partial(
        _build_spline,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        periodic=periodic,
        linear_tails=linear_tails,
        min_width=min_width,
        min_height=min_height,
        min_deriv=min_deriv,
    )

    in_dims = (0, 0) if params_are_batched else (0, None)
    vmap = partial(torch.vmap, in_dims=in_dims, out_dims=(0, 0))

    wrapper = partial(
        _extend_to_reals if linear_tails else _clamp_to_interval,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        tol=bounds_tol,
    )
    forward_fn = vmap(wrapper(_forward_transform))
    inverse_fn = vmap(wrapper(_inverse_transform))

    class RQSplineTransform:
        def __init__(self, params: Tensor):
            self.params = build_spline(params)

        def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
            return forward_fn(x, self.params)

        def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
            return inverse_fn(y, self.params)

    return RQSplineTransform


class RQSplineModule(TransformModule):
    def __init__(
        self,
        *,
        n_segments: int,
        net_hidden_shape: Optional[list[int]] = None,
        net_activation: Optional[str] = None,
        min_width: float = 1e-3,
        min_height: float = 1e-3,
        min_deriv: float = 1e-3,
        bounds_tol: float = 1e-4,
    ):
        super().__init__(
            n_params=3 * n_segments - 1,
            net_hidden_shape=net_hidden_shape,
            net_activation=net_activation,
        )
        self._transform = rqspline_transform(
            params_are_batched=net_hidden_shape is not None,
            lower_bound=-1.0,
            upper_bound=+1.0,
            linear_tails=True,
            min_width=min_width,
            min_height=min_height,
            min_deriv=min_deriv,
            bounds_tol=bounds_tol,
        )

    def transform(self, params: Tensor) -> Transform:
        return self._transform(params)


class CircularSplineModule(CircularTransformModule):
    def __init__(
        self,
        *,
        n_segments: int,
        net_hidden_shape: Optional[list[int]] = None,
        net_activation: Optional[str] = None,
        min_width: float = 1e-3,
        min_height: float = 1e-3,
        min_deriv: float = 1e-3,
        bounds_tol: float = 1e-4,
    ):
        super().__init__(
            n_params=3 * n_segments,
            net_hidden_shape=net_hidden_shape,
            net_activation=net_activation,
        )
        self._transform = rqspline_transform(
            params_are_batched=net_hidden_shape is not None,
            lower_bound=0.0,
            upper_bound=2 * π,
            periodic=True,
            min_width=min_width,
            min_height=min_height,
            min_deriv=min_deriv,
            bounds_tol=bounds_tol,
        )

    def transform(self, params: Tensor) -> Transform:
        return self._transform(params)
