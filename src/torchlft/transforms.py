from abc import ABC, abstractmethod
from math import pi as π
import math

import torch
import torch.linalg as LA
import torch.nn.functional as F

import torchlft.constraints as constraints
from torchlft.utils.tensor import (
    sum_except_batch,
    mod_2pi,
    expand_like_stack,
)
from torchlft.utils.linalg import dot

from torchlft.typing import Constraint, Tensor, BoolTensor

DEBUG = False

# NOTE: unsure whether to support giving a single parameter, or one per batch,
# rather than for each lattice site.
# May lead to hard to catch bugs


class Translation:
    domain: Constraint = constraints.real
    arg_constraints: dict[str, Constraint] = {"shift": constraints.real}
    n_params: int = 1

    def __init__(self, shift: Tensor) -> None:
        self.shift = shift

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        t = self.shift.expand_as(x)
        y = x + t
        ldj = torch.zeros(x.shape[0], device=x.device)
        return y, ldj

    def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
        t = self.shift.expand_as(y)
        x = y - t
        ldj = torch.zeros(y.shape[0], device=y.device)
        return x, ldj


class Rescaling:
    domain: Constraint = constraints.real
    arg_constraints: dict[str, Constraint] = {"log_scale": constraints.real}
    pointwise: bool = True
    n_params: int = 1

    def __init__(self, log_scale: Tensor) -> None:
        self.log_scale = log_scale

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        s = self.log_scale.expand_as(x)
        y = x * torch.exp(s)
        ldj = sum_except_batch(s)
        return y, ldj

    def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
        s = self.log_scale.expand_as(y)
        x = y * torch.exp(-s)
        ldj = sum_except_batch(-s)
        return x, ldj


class AffineTransform:
    domain: Constraint = constraints.real
    arg_constraints: dict[str, Constraint] = {
        "log_scale": constraints.real,
        "shift": constraints.real,
    }
    pointwise = True
    n_params = 2

    def __init__(self, log_scale: Tensor, shift: Tensor) -> None:
        self.log_scale = log_scale
        self.shift = shift

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        s, t = self.log_scale.expand_as(x), self.shift.expand_as(x)
        y = x * torch.exp(s) + t
        ldj = sum_except_batch(s)
        return y, ldj

    def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
        s, t = self.log_scale.expand_as(y), self.shift.expand_as(y)
        x = (y - t) * torch.exp(-s)
        ldj = sum_except_batch(self.log_scale).negative()
        return x, ldj


class _RQSplineTransform:
    def __init__(
            self,
            knots_x: Tensor,
            knots_y: Tensor,
            knots_dydx: Tensor,
    ):
        self.knots_x = knots_x
        self.knots_y = knots_y
        self.knots_dydx = knots_dydx

    def _get_segment(
        self, inputs: Tensor, inverse: bool
    ) -> tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, BoolTensor
    ]:
        outside_bounds_mask = (inputs < self._lower_bound) | (
            inputs > self._upper_bound
        )
        self.handle_inputs_outside_bounds(inputs, outside_bounds_mask)

        knots = self.knots_y if inverse else self.knots_x

        i0 = (torch.searchsorted(knots, inputs.unsqueeze(-1)) - 1).clamp_(
            0, self._n_segments - 1
        )
        i0_i1 = torch.stack((i0, i0 + 1), dim=0)

        x0_x1 = (
            expand_like_stack(self.knots_x, 2).gather(-1, i0_i1).squeeze(-1)
        )
        y0_y1 = (
            expand_like_stack(self.knots_y, 2).gather(-1, i0_i1).squeeze(-1)
        )
        d0_d1 = (
            expand_like_stack(self.knots_dydx, 2).gather(-1, i0_i1).squeeze(-1)
        )

        # NOTE: Cannot do x0, x1 = x0_x1 with torchscript :(
        x0, x1 = x0_x1[0], x0_x1[1]
        y0, y1 = y0_y1[0], y0_y1[1]
        d0, d1 = d0_d1[0], d0_d1[1]

        s = (y1 - y0) / (x1 - x0)

        return x0, x1, y0, y1, d0, d1, s, outside_bounds_mask

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # assert list(x.shape) == self.knots_x.shape[:-1]

        x0, x1, y0, y1, d0, d1, s, outside_bounds_mask = self._get_segment(
            x, inverse=False
        )

        θx = (x - x0) / (x1 - x0)

        denominator = s + (d1 + d0 - 2 * s) * θx * (1 - θx)

        θy = (s * θx**2 + d0 * θx * (1 - θx)) / denominator

        y = y0 + (y1 - y0) * θy

        dydx = (
            s**2
            * (d1 * θx**2 + 2 * s * θx * (1 - θx) + d0 * (1 - θx) ** 2)
            / denominator**2
        )
        # assert torch.all(dydx > 0)

        y[outside_bounds_mask] = x[outside_bounds_mask]
        # NOTE: this shouldn't be necessary! Should be 1 by construction
        dydx[outside_bounds_mask] = 1

        ldj = sum_except_batch(dydx.log())

        return y, ldj

    def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
        # assert list(y.shape) == self.knots_y.shape[:-1]

        x0, x1, y0, y1, d0, d1, s, outside_bounds_mask = self._get_segment(
            y, inverse=True
        )

        θy = (y - y0) / (y1 - y0)

        b = d0 - (d1 + d0 - 2 * s) * θy
        a = s - b
        c = -s * θy

        θx = (-2 * c) / (b + (b**2 - 4 * a * c).sqrt())

        x = x0 + (x1 - x0) * θx

        denominator = s + (d1 + d0 - 2 * s) * θx * (1 - θx)

        dydx = (
            s**2
            * (d1 * θx**2 + 2 * s * θx * (1 - θx) + d0 * (1 - θx) ** 2)
            / denominator**2
        )

        x[outside_bounds_mask] = y[outside_bounds_mask]
        dydx[outside_bounds_mask] = 1

        ldj = sum_except_batch(dydx.log()).negative()

        return x, ldj


class RQSplineTransform:
    """
    This uses the parametrisation introduced by Gregory and Delbourgo
    (1983)

    NOTE: segment dimension has to be last one (searchsorted)

    NOTE: don't instantiate this directly. See one of the three versions

    Args:
        widths
            Un-normalised segment sizes in the domain
        heights
            Un-normalised segment sizes in the codomain
        derivs
            Unconstrained derivatives at the knots

    Returns:
        Tuple of tensors containing (1) Normalised segment sizes in
        the domain, (2) Normalised segment sizes in the codomain, (3)
        Constrained derivatives at the knots, (4) Coordinates of the
        knots in the domain, (5) Coordinates of the knots in the codomain


    References:
        Gregory, J. A. & Delbourgo, R. C2 Rational \
        Quadratic Spline Interpolation to Monotonic Data, IMA Journal of \
        Numerical Analysis, 1983, 3, 141-152
        """

    pointwise = True

    def __init__(
        self,
        widths: Tensor,
        heights: Tensor,
        derivs: Tensor,
        *,
        lower_bound: float,
        upper_bound: float,
        bounded: bool,
        periodic: bool,
        min_slope: float,
    ) -> None:
        assert not (periodic and not bounded)

        assert widths.shape == heights.shape

        # Normalise the widths and heights to the interval
        widths = F.softmax(widths, dim=-1) * (upper_bound - lower_bound)
        heights = F.softmax(heights, dim=-1) * (upper_bound - lower_bound)

        # Ensure the derivatives are positive and > min_slope
        derivs = F.softplus(derivs) + min_slope

        """
        if DEBUG:
            constraint = constraints.positive + constraints.SumToValue(
                upper_bound - lower_bound, dim=-1
            )
            constraint.check(widths)
            constraint.check(heights)
            constraints.positive.check(derivs)
        """

        # Apply boundary conditions to the derivatives
        if not bounded:
            # match derivs with identity transform outside bounds
            derivs = F.pad(derivs, (1, 1), "constant", 1.0)
        elif periodic:
            # match derivs at 0 and 2pi
            derivs = F.pad(
                derivs.flatten(1, -2), (0, 1), "circular"
            ).unflatten(1, derivs.shape[1:-1])
        else:
            # bounded and not periodic: no additional constraints
            derivs = derivs

        assert derivs.shape[-1] == widths.shape[-1] + 1

        zeros = torch.zeros(
            size=widths.shape[:-1],
            device=widths.device,
            dtype=widths.dtype,
        ).unsqueeze(-1)

        # Build the spline
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

        # We only need these three tensors to compute the transformation
        self.knots_x = knots_x
        self.knots_y = knots_y
        self.knots_dydx = derivs

        self._n_segments = widths.shape[-1]
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._bounded = bounded
        self._periodic = periodic

    @property
    def n_segments(self) -> int:
        return self._n_segments

    @property
    def upper_bound(self) -> float:
        return self._upper_bound

    @property
    def lower_bound(self) -> float:
        return self._lower_bound

    @property
    def bounded(self) -> bool:
        return self._bounded

    @property
    def periodic(self) -> bool:
        return self._periodic

    # @property
    # def domain(self) -> Constraint:
    #    return (
    #        constraints.OpenInterval(self.lower_bound, self.upper_bound)
    #        if self.bounded
    #        else constraints.real
    #    )

    def __call__(self, widths: Tensor, heights: Tensor, derivs: Tensor) -> Callable[Tensor, [Tensor, Tensor]]:
        assert widths.shape == heights.shape

        # Normalise the widths and heights to the interval
        widths = F.softmax(widths, dim=-1) * (self.upper_bound - self.lower_bound)
        heights = F.softmax(heights, dim=-1) * (self.upper_bound - self.lower_bound)

        # Ensure the derivatives are positive and > min_slope
        derivs = F.softplus(derivs) + self.min_slope

        """
        if DEBUG:
            constraint = constraints.positive + constraints.SumToValue(
                upper_bound - lower_bound, dim=-1
            )
            constraint.check(widths)
            constraint.check(heights)
            constraints.positive.check(derivs)
        """

        # Apply boundary conditions to the derivatives
        if not self.bounded:
            # match derivs with identity transform outside bounds
            derivs = F.pad(derivs, (1, 1), "constant", 1.0)
        elif self.periodic:
            # match derivs at 0 and 2pi
            derivs = F.pad(
                derivs.flatten(1, -2), (0, 1), "circular"
            ).unflatten(1, derivs.shape[1:-1])
        else:
            # bounded and not periodic: no additional constraints
            derivs = derivs

        assert derivs.shape[-1] == widths.shape[-1] + 1

        zeros = torch.zeros(
            size=widths.shape[:-1],
            device=widths.device,
            dtype=widths.dtype,
        ).unsqueeze(-1)

        # Build the spline
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

        return _RationalQuadraticSpline(knots_x, knots_y, knots_dydx)

        # We only need these three tensors to compute the transformation
        self.knots_x = knots_x
        self.knots_y = knots_y
        self.knots_dydx = derivs

        self._n_segments = widths.shape[-1]
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._bounded = bounded
        self._periodic = periodic

    def handle_inputs_outside_bounds(
        self, inputs: Tensor, outside_bounds_mask: BoolTensor
    ) -> None:
        """
        Handle inputs falling outside the spline bounds.

        By default, this method has two distinct behaviour depending on
        whether the ``bounded`` attribute is true or false. If it is true,
        any inputs falling outside the bounds causes an exception to be
        raised. If it is false, this method submits a :code:`log.info` logging
        event which details the fraction of inputs falling outside the bounds.

        Args:
            inputs
                Tensor of inputs to the transformation
            outside_bounds_mask
                BoolTensor of the same shape as the layer input where the
                :code:`True` elements correspond to inputs which fell outside
                the spline bounds.

        """
        if self._bounded:
            if outside_bounds_mask.any():
                raise Exception  # custom exception needed
        else:
            pass  # TODO: log.info



class CircularSplineTransform(RQSplineTransform):
    def __init__(
        self,
        widths: Tensor,
        heights: Tensor,
        derivs: Tensor,
        *,
        min_slope: float = 1e-3,
    ) -> None:
        super().__init__(
            widths,
            heights,
            derivs,
            lower_bound=0,
            upper_bound=2 * π,
            bounded=True,
            periodic=True,
            min_slope=min_slope,
        )


class IntegratedBSplineTransform:
    def __init__(
        self,
        intervals: Tensor,
        weights: Tensor,
        *,
        lower_bound: float,
        upper_bound: float,
        min_interval: float = 1e-1,
        min_weight: float = 1e-3,
    ):
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

        intervals = torch.sigmoid(intervals) + min_interval
        weights = (
            F.softplus(weights) + min_weight
        )  # min weight TODO make configurable

        Δ, ρ = intervals, weights

        Δ = Δ / Δ.sum(dim=-1, keepdim=True)
        ρ = ρ / (((ρ[..., :-2] + ρ[..., 1:-1] + ρ[..., 2:]) / 3) * Δ).sum(
            dim=-1, keepdim=True
        )

        Δpad = F.pad(Δ, (1, 1), "constant", 0)

        ω = (ρ[..., 1:] - ρ[..., :-1]) / (Δpad[..., :-1] + Δpad[..., 1:])
        h = ρ[..., 1:-1] * Δ + (1 / 3) * (ω[..., 1:] - ω[..., :-1]) * Δ**2

        zeros = torch.zeros(
            size=(*Δ.shape[:-1], 1),
            device=Δ.device,
            dtype=Δ.dtype,
        )
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

        self.intervals = Δpad
        self.weights = ρ
        self.omega = ω
        self.knots_x = knots_x
        self.knots_y = knots_y

    @property
    def upper_bound(self) -> float:
        return self._upper_bound

    @property
    def lower_bound(self) -> float:
        return self._lower_bound

    @property
    def domain(self) -> Constraint:
        return constraints.OpenInterval(self._lower_bound, self._upper_bound)

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = (x - self.lower_bound) / (self.upper_bound - self.lower_bound)

        i = torch.searchsorted(self.knots_x, x, side="right").clamp(
            1, self.n_segments
        )

        # Get parameters of the segments that x falls in
        Δ = torch.gather(self.intervals, -1, i)
        ρ = torch.gather(self.weights, -1, i)
        ωi = torch.gather(self.omega, -1, i)
        ωim1 = torch.gather(self.omega, -1, i - 1)
        x0 = torch.gather(self.knots_x, -1, i - 1)
        y0 = torch.gather(self.knots_y, -1, i - 1)

        θ = (x - x0) / Δ

        y = (
            y0
            + ρ * Δ * θ
            - ωim1 * Δ**2 * θ * (1 - θ)
            + (1 / 3) * (ωi - ωim1) * Δ**2 * θ**3
        )

        dydx = ρ + ωi * Δ * θ**2 - ωim1 * Δ * (1 - θ) ** 2

        y = y * (self.upper_bound - self.lower_bound) + self.lower_bound

        ldj = sum_except_batch(dydx.log())

        return y, ldj

    def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class MobiusTransform:
    domain: Constraint = constraints.UnitNorm(dim=-1)
    codomain: Constraint = constraints.UnitNorm(dim=-1)
    arg_constraints: {"omega": constraints.UnitBall(dim=-1)}

    def __init__(self, omega: Tensor, *, epsilon: float = 1e-3):
        # NOTE: perhaps constraining omega should occur within this
        # constructor, so unconstrained inputs can be passed in
        self.omega = omega

    @staticmethod
    def identity_params(x: Tensor) -> dict[str, Tensor]:
        return {"omega": torch.zeros_like(x)}

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        ω = self.omega
        dydx = ((1 - dot(ω, ω)) / dot(x - ω, x - ω)).unsqueeze(-1)
        y = dydx * (x - ω) - ω
        ldj = sum_except_batch(dydx.log())
        return y, ldj

    def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
        ω = self.omega
        dxdy = ((1 - dot(ω, ω)) / dot(y + ω, y + ω)).unsqueeze(-1)
        x = dxdy * (y + ω) + ω
        ldj = sum_except_batch(dxdy.log())
        return x, ldj


class MobiusMixtureTransform:
    domain: Constraint = constraints.UnitNorm(dim=-1)
    codomain: Constraint = constraints.UnitNorm(dim=-1)
    arg_constraints: {
        "omega": constraints.UnitBall(dim=-1),
        "weights": constraints.real,
    }

    def __init__(
        self,
        omega: Tensor,
        weights: Tensor | None = None,
        *,
        epsilon: float = 1e-3,
    ):
        # TODO replace with constraint to disk
        assert LA.vector_norm(omega, dim=-1) < 1
        assert omega.shape[-1] == 2  # only valid for circle

        n_mixture = omega.shape[-2]

        if weights is not None:
            assert weights.shape[-1] == n_mixture
            weights = torch.softmax(weights, dim=-1)
        else:
            weights = 1 / n_mixture

        self._n_mixture = n_mixture
        self.omega = omega
        self.weights = weights

    @property
    def n_mixture(self) -> int:
        return self._n_mixture

    @staticmethod
    def identity_params(
        x: Tensor, n_mixture: int, weighted: bool
    ) -> dict[str, Tensor]:
        # NOTE: torchlft.utils.tensor.dot expects the vector dim to be -1,
        # which forces us to use dim -2 for the mixture components.
        omega = torch.zeros(
            *x.shape[:-1], n_mixture, 2, device=x.device, dtype=x.dtype
        )
        weights = (
            torch.full(
                (*x.shape[:-1], n_mixture),
                (1 / n_mixture),
                device=x.device,
                dtype=x.dtype,
            )
            if weighted
            else None
        )
        return {"omega": omega, "weights": weights}

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        ω, ρ = self.omega, self.weights

        x0 = torch.tensor([1, 0]).type_as(x).expand_as(x)
        x_x0 = torch.stack([x, x0], dim=0).unsqueeze(-2)

        dydx = ((1 - dot(ω, ω)) / dot(x_x0 - ω, x_x0 - ω)).unsqueeze(-1)
        y_y0 = dydx * (x_x0 - ω) - ω

        # Now rotate s.t (1, 0) -> (1, 0)
        ϕ, ϕ0 = torch.atan2(*reversed(y_y0.split(1, dim=-1)))
        ϕ = mod_2pi(ϕ - ϕ0)

        Σρϕ = (ρ * ϕ).sum(dim=-2)
        y = torch.cat([torch.cos(Σρϕ), torch.sin(Σρϕ)], dim=-1)

        dydx, _ = dydx
        dydx = (ρ * dydx).sum(dim=-2)
        ldj = sum_except_batch(dydx.log())

        return y, ldj

    def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class ProjectedAffineTransform:
    domain: constraints.periodic
    codomain: constraints.periodic
    arg_constraints: {"log_scale": constraints.real, "shift": constraints.real}

    def __init__(
        self,
        log_scale: Tensor,
        shift: Tensor | None,
        *,
        linear_thresh: float | None = None,
    ):
        self.log_scale = log_scale
        self.shift = shift if shift is not None else 0
        self._linear_thresh = linear_thresh

    @staticmethod
    def identity_params(x: Tensor, shift: bool) -> dict[str, Tensor]:
        return {
            "log_scale": torch.zeros_like(x),
            "shift": torch.zeros_like(x) if shift else None,
        }

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        log_α, β = self.log_scale, self.shift
        α = torch.exp(log_α)

        y = mod_2pi(2 * torch.atan(α * torch.tan((x - π) / 2) + β) + π)

        dxdy = (
            (1 + β**2) / α * torch.sin(x / 2) ** 2
            + α * torch.cos(x / 2) ** 2
            - β * torch.sin(x)
        )

        if self._linear_thresh is None:
            ldj = sum_except_batch(dxdy.log().negative())
            return y, ldj

        m1 = x < self._linear_thresh
        m2 = (2 * π - x) < self._linear_thresh

        y[m1] = (x / α)[m1]
        y[m2] = (2 * π - (2 * π - x) / α)[m2]
        dxdy[m1 | m2] = α[m1 | m2]

        ldj = sum_except_batch(dxdy.log().negative())

        return y, ldj

    def inverse(self, y: Tensor) -> Tensor:
        log_α, β = self.log_scale, self.shift
        α = torch.exp(log_α)

        x = mod_2pi(2 * torch.atan((1 / α) * torch.tan((y - π) / 2) - β)) + π

        dxdy = (
            (1 + β**2) / α * torch.sin(x / 2) ** 2
            + α * torch.cos(x / 2) ** 2
            - β * torch.sin(x)
        )

        if self._linear_thresh is None:
            ldj = sum_except_batch(dxdy.log())
            return x, ldj

        m1 = y < self._linear_thresh
        m2 = (2 * π - y) < self._linear_thresh

        y[m1] = (y * α)[m1]
        y[m2] = (2 * π - (2 * π - y) * α)[m2]
        dxdy[m1 | m2] = α[m1 | m2]

        ldj = sum_except_batch(dxdy.log())

        return x, ldj


class ProjectedAffineMixtureTransform:
    domain: constraints.periodic
    codomain: constraints.periodic
    arg_constraints: {
        "log_scale": constraints.real,
        "shift": constraints.real,
        "weights": constraints.real,
    }

    def __init__(
        self,
        log_scale: Tensor,
        shift: Tensor | None,
        weights: Tensor | None,
        *,
        linear_thresh: float | None = None,
    ):
        n_mixture = log_scale.shape[-1]

        if weights is not None:
            assert weights.shape[-1] == n_mixture
            weights = torch.softmax(weights, dim=-1)
        else:
            weights = 1 / n_mixture

        self._n_mixture = n_mixture
        self.log_scale = log_scale
        self.shift = shift if shift is not None else 0
        self.weights = weights

        self._linear_thresh = linear_thresh

    @staticmethod
    def identity_params(
        x: Tensor, n_mixture: int, use_shift: bool, weighted: bool
    ) -> dict[str, Tensor]:
        shape = (*x.shape, n_mixture)
        log_scale = torch.zeros(shape).type_as(x)
        shift = torch.zeros(shape).type_as(x) if use_shift else None
        weights = (
            torch.full(shape, (1 / n_mixture)).type_as(x) if weighted else None
        )
        return {"log_scale": log_scale, "shift": shift, "weights": weights}

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        log_α, β, ρ = self.log_scale, self.shift, self.weights
        α = torch.exp(log_α)
        x = x.unsqueeze(-1)  # mixture dimension

        y = mod_2pi(2 * torch.atan(α * torch.tan((x - π) / 2) + β) + π)

        dxdy = (
            (1 + β**2) / α * torch.sin(x / 2) ** 2
            + α * torch.cos(x / 2) ** 2
            - β * torch.sin(x)
        )

        if self._linear_thresh is not None:
            m1 = x < self._linear_thresh
            m2 = (2 * π - x) < self._linear_thresh

            y[m1] = (x / α)[m1]
            y[m2] = (2 * π - (2 * π - x) / α)[m2]
            dxdy[m1 | m2] = α[m1 | m2]

        y = (ρ * y).sum(dim=-1)
        dydx = (ρ * (1 / dxdy)).sum(dim=-1)

        ldj = sum_except_batch(dydx.log())
        return y, ldj

    def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError
