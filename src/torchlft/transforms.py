from math import exp, log, pi as π

import torch

from torchlft.abc import Constraint, Transform
import torchlft.constraints as constraints
from torchlft.utils.tensor import sum_except_batch, mod_2pi
from torchlft.typing import *


class Translation(Transform):
    domain: Constraint = constraints.real
    codomain: Constraint = constraints.real
    param_constraints: dict[str, Constraint] = {"shift": constraints.real}

    def __init__(self, shift: Tensor) -> None:
        self.shift = shift

    @staticmethod
    def identity_params(x: Tensor) -> dict[str, Tensor]:
        return {"shift": torch.zeros_like(x)}

    def forward(self, x: Tensor) -> Tensor:
        return x + self.shift

    def inverse(self, y: Tensor) -> Tensor:
        return y - self.shift

    def log_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.zeros(x.shape[0], device=x.device)


class Rescaling(Transform):
    domain: Constraint = constraints.real
    codomain: Constraint = constraints.real
    param_constraints: dict[str, Constraint] = {"log_scale": constraints.real}

    def __init__(self, log_scale: Tensor) -> None:
        self.log_scale = log_scale

    @staticmethod
    def identity_params(x: Tensor) -> dict[str, Tensor]:
        return {"log_scale": torch.zeros_like(x)}

    def forward(self, x: Tensor) -> Tensor:
        return x * self.log_scale.negative().exp()

    def inverse(self, y: Tensor) -> Tensor:
        return y * self.log_scale.exp()

    def log_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return sum_except_batch(self.log_scale)


class AffineTransform(Transform):
    domain: Constraint = constraints.real
    codomain: Constraint = constraints.real
    param_constraints: dict[str, Constraint] = {
        "log_scale": constraints.real,
        "shift": constraints.real,
    }

    def __init__(self, log_scale: Tensor, shift: Tensor) -> None:
        self.log_scale = log_scale
        self.shift = shift

    @staticmethod
    def identity_params(x: Tensor) -> Tensor:
        return {"log_scale": torch.zeros_like(x), "shift": torch.zeros_like(x)}

    def forward(self, x: Tensor) -> Tensor:
        s, t = self.log_scale, self.shift
        return (x + t) * torch.exp(-s)

    def inverse(self, y: Tensor) -> Tensor:
        s, t = self.log_scale, self.shift
        return y * torch.exp(s) - t

    def log_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return sum_except_batch(self.log_scale)


class _RQSplineTransform(Transform):
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

    param_constraints: dict[str, Constraint] = {
        "widths": constraints.real,
        "heights": constraints.real,
        "derivs": constraints.real,
    }

    def __init__(
        self,
        widths: Tensor,
        heights: Tensor,
        derivs: Tensor,
        *,
        lower_bound: float,
        upper_bound: float,
        min_slope: float = 1e-3,
    ) -> None:

        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

        self._n_segments = widths.shape[-1]
        assert heights.shape[-1] == self._n_segments

        # Normalise the widths and heights to the interval
        interval_size = upper_bound - lower_bound
        widths = interval_size * F.softmax(widths, dim=-1)
        heights = interval_size * F.softmax(heights, dim=-1)

        # Ensure the derivatives are positive and > min_slope
        derivs = F.softplus(derivs) + min_slope

        # Apply boundary conditions to the derivatives
        if self.domain is constraints.real:
            derivs = F.pad(derivs, (1, 1), "constant", 1)  # linear tails
        elif self.domain is constraints.periodic:
            derivs = F.pad(derivs.flatten(1, -2), (0, 1), "circular").view(
                *derivs.shape[:-1], -1
            )  # match derivs at 0 and 2pi
        else:
            derivs = derivs  # no additional constraints

        assert derivs.shape[-1] == self._n_segments + 1

        zeros = torch.zeros(
            size=(*widths.shape[:-1], 1),
            device=widths.device,
            dtype=widths.dtype,
        )

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

    @property
    def n_segments(self) -> int:
        return self._n_segments

    @property
    def upper_bound(self) -> float:
        return self._upper_bound

    @property
    def lower_bound(self) -> float:
        return self._lower_bound

    @staticmethod
    def identity_params(x: Tensor, n_segments: int) -> dict[str, Any]:
        widths = (
            torch.full_like(x, fill_value=(1 / n_segments))
            .unsqueeze(-1)
            .repeat([1 for _ in x.shape] + [n_segments])
        )
        derivs = (
            torch.full_like(x, fill_value=log(exp(1) - 1))
            .unsqueeze(-1)
            .repeat([1 for _ in x.shape] + [n_segments - 1])
        )
        return {
            "widths": widths,
            "heights": widths.clone(),
            "derivs": derivs,
            "min_slope": 0,
        }

    def handle_inputs_outside_bounds(
        self, inputs: Tensor, outside_bounds_mask: BoolTensor
    ) -> None:
        """
        Handle inputs falling outside the spline interval.

        Unless overridden, this method submits a :code:`log.debug` logging
        event if more than 1/1000 inputs fall outside the spline interval.

        Args:
            inputs
                Tensor of inputs to the transformation
            outside_bounds_mask
                BoolTensor of the same shape as the layer input where the
                :code:`True` elements correspond to inputs which fell outside
                the spline bounds.

        """
        pass

    def _get_segment(
        self, inputs: Tensor, inverse: bool = False
    ) -> tuple[Tensor]:
        outside_bounds_mask = (inputs < self._lower_bound) | (
            inputs > self._upper_bound
        )
        self.handle_inputs_outside_bounds(inputs, outside_bounds_mask)

        knots = self.knots_y if inverse else self.knots_x

        i0 = (torch.searchsorted(knots, inputs.unsqueeze(-1)) - 1).clamp_(
            0, self._n_segments - 1
        )
        i0_i1 = torch.stack((i0, i0 + 1), dim=0)

        x0, x1 = self.knots_x.gather(-1, i0_i1).squeeze(-1)
        y0, y1 = self.knots_y.gather(-1, i0_i1).squeeze(-1)
        d0, d1 = self.knots_dydx.gather(-1, i0_i1).squeeze(-1)

        s = (y1 - y0) / (x1 - x0)

        return x0, x1, y0, y1, d0, d1, s, outside_bounds_mask

    def _forward_and_gradient(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # assert list(x.shape) == self.knots_x.shape[:-1]

        x0, x1, y0, y1, d0, d1, s, outside_bounds_mask = self._get_segment(x)

        θx = (x - x0) / (x1 - x0)

        denominator_recip = (
            s + (d1 + d0 - 2 * s) * θx * (1 - θx)
        ).reciprocal()

        θy = (s * θx**2 + d0 * θx * (1 - θx)) * denominator_recip

        y = y0 + (y1 - y0) * θy

        dydx = (
            s**2
            * (d1 * θx**2 + 2 * s * θx * (1 - θx) + d0 * (1 - θx) ** 2)
            * denominator_recip**2
        )
        # assert torch.all(dydx > 0)

        y[outside_interval_mask] = x[outside_interval_mask]
        # NOTE: this shouldn't be necessary! Should be 1 by construction
        dydx[outside_interval_mask] = 1

        return y, dydx

    def _inverse_and_gradient(self, y: Tensor) -> Tensor:
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

        denominator_recip = (
            s + (d1 + d0 - 2 * s) * alpha * (1 - alpha)
        ).reciprocal()

        dydx = (
            s**2
            * (d1 * θx**2 + 2 * s * θx * (1 - θx) + d0 * (1 - θx) ** 2)
            * denominator_recip**2
        )
        assert torch.all(dydx > 0)

        x[outside_interval_mask] = y[outside_interval_mask]
        dydx[outside_interval_mask] = 1

        return x, dydx

    def forward(self, x: Tensor) -> Tensor:
        y, _ = self._forward_and_gradient(x)
        return y

    def inverse(self, y: Tensor) -> Tensor:
        x, _ = self._inverse_and_gradient(y)
        return x

    def log_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        _, dydx = self._forward_and_gradient(x)
        return sum_except_batch(dydx.log())

    def forward_and_ldj(self, x: Tensor) -> Tensor:
        y, dydx = self._forward_and_gradient(x)
        ldj = sum_except_batch(dydx.log())
        return y, ldj

    def inverse_and_ldj(self, y: Tensor) -> Tensor:
        x, dydx = self._inverse_and_gradient(x)
        ldj = sum_except_batch(dydx.log().negative())
        return x, ldj


class RQSplineTransform(_RQSplineTransform):
    domain: Constraint = constraints.real
    codomain: Constraint = constraints.real


class CircularRQSplineTransform(_RQSplineTransform):
    domain: Constraint = constraints.periodic
    codomain: Constraint = constraints.periodic

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
            min_slope=min_slope,
        )

    def handle_inputs_outside_bounds(
        self, inputs: Tensor, outside_bounds_mask: BoolTensor
    ) -> None:
        if outside_bounds_mask.any():
            raise Exception  # custom exception needed


class BoundedRQSplineTransform(_RQSplineTransform):
    @property
    def domain(self) -> Constraint:
        return OpenInterval(self.lower_bound, self._upper_bound)

    @property
    def codomain(self) -> Constraint:
        return OpenInterval(self._lower_bound, self._upper_bound)

    def handle_inputs_outside_bounds(
        self, inputs: Tensor, outside_bounds_mask: BoolTensor
    ) -> None:
        if outside_bounds_mask.any():
            raise Exception


class IntegratedBSplineTransform(Transform):

    def __init__(self, intervals: Tensor, weights: Tensor, *, lower_bound: float, upper_bound: float, min_interval: float = 1e-1, min_weight: float = 1e-3):

        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

        intervals = torch.sigmoid(intervals) + min_interval
        weights = F.softplus(weights) + min_weight  # min weight TODO make configurable

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
            dtype=Δs.dtype,
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
        return OpenInterval(self.lower_bound, self._upper_bound)

    @property
    def codomain(self) -> Constraint:
        return OpenInterval(self._lower_bound, self._upper_bound)
    
    @staticmethod
    def identity_params(x: Tensor) -> dict[str, Tensor]:
        raise NotImplementedError # TODO

    def _forward_and_gradient(self, x: Tensor) -> tuple[Tensor, Tensor]:

        x = (x - self.lower_bound) / (
            self.upper_bound - self.lower_bound
        )

        i = torch.searchsorted(self.knots_x, x, side="right").clamp(1, self.n_segments)

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

        ldj = dydx.log().flatten(start_dim=1).sum(dim=1)

        y = (
            y * (self.upper_bound - self.lower_bound)
            + self.lower_bound
        )

        return y, dydx

    def forward(self, x: Tensor) -> Tensor:
        y, _ = self._forward_and_gradient(x)
        return y

    def inverse(self, y: Tensor) -> Tensor:
        raise NotImplementedError

    def log_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        _, dydx = self._forward_and_gradient(x)
        return sum_except_batch(dydx.log())

    def forward_and_ldj(self, x: Tensor) -> tuple[Tensor, Tensor]:
        y, dydx = self._forward_and_gradient(x)
        return y, sum_except_batch(dydx.log())

    def inverse_and_ldj(self, y: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

class MobiusTransform(Transform):
    domain: Constraint = constraints.unit_norm
    codomain: Constraint = constraints.unit_norm
    arg_constraints: {"omega": None} # TODO
    
    def __init__(self, omega: Tensor, *, epsilon: float = 1e-3):
        # TODO replace with constraint to disk
        assert LA.vector_norm(omega, dim=-1) < 1
        
        self.omega = omega

    @staticmethod
    def identity_params(x: Tensor) -> dict[str, Tensor]:
        return {"omega": torch.zeros_like(x)}

    def forward_and_ldj(self, x: Tensor) -> Tensor:
        ω = self.omega
        x_minus_ω = x - ω
        
        dydx = ((1 - dot(ω, ω)) / dot(x_minus_ω, x_minus_ω)).unsqueeze(-1)
        y = dydx * x_minus_ω - ω

        ldj = sum_except_batch(dydx.log())

        return y, ldj

    def forward(self, x: Tensor) -> Tensor:
        y, _ = self.forward_and_ldj(x)
        return y

    def inverse(self, y: Tensor) -> Tensor:
        raise NotImplementedError
    
    def inverse_and_ldj(self, y: Tensor) -> Tensor:
        raise NotImplementedError

    def log_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        ω = self.omega
        x_minus_ω = x - ω
        coeff = ((1 - dot(ω, ω)) / dot(x_minus_ω, x_minus_ω))
        dydx = (ρ * coeff).sum(dim=-2)
        return sum_except_batch(dydx.log())

class MobiusMixtureTransform(Transform):
    domain: Constraint = constraints.unit_norm
    codomain: Constraint = constraints.unit_norm

    def __init__(self, omega: Tensor, weights: Tensor | None = None, *, epsilon: float = 1e-3):

        n_mixture = omega.shape[-2]
        
        assert LA.vector_norm(omega, dim=-1) < 1

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
    def identity_params(x: Tensor) -> dict[str, Tensor]:
        return {"omega": torch.zeros_like(x), "weights": None}

    def forward_and_ldj(self, x: Tensor) -> Tensor:
        ω, ρ = self.omega, self.weights
        
        x_10 = torch.tensor([1, 0]).type_as(x).expand_as(x)

        x = torch.stack(
                [x, x_10],
                dim=0
        ).unsqueeze(-2)

        x_minus_ω = x - ω
        
        coeff = ((1 - dot(ω, ω)) / dot(x_minus_ω, x_minus_ω)).unsqueeze(-1)
        fx = coeff * x_minus_ω - ω

        θ, θ_10 = torch.atan2(*reversed(fx.split(1, dim=-1)))
        θ = (ρ * mod_2pi(θ - θ_10)).sum(dim=-2)
        
        y = torch.cat([θ.cos(), θ.sin()], dim=-1)

        ldj = sum_except_batch((ρ * coeff[0]).sum(dim=-2).log())

        return y, ldj

    def forward(self, x: Tensor) -> Tensor:
        y, _ = self.forward_and_ldj(x)
        return y

    def inverse(self, y: Tensor) -> Tensor:
        raise NotImplementedError
    
    def inverse_and_ldj(self, y: Tensor) -> Tensor:
        raise NotImplementedError

    def log_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        ω, ρ = self.omega, self.weights
        x_minus_ω = x - ω
        coeff = ((1 - dot(ω, ω)) / dot(x_minus_ω, x_minus_ω))
        dydx = (ρ * coeff).sum(dim=-2)
        return sum_except_batch(dydx.log())


class ProjectedAffineTransform(Transform):
    domain: periodic
    codomain: periodic

    def __init__(self, log_scale: Tensor, shift: Tensor | None, weights: Tensor | None, linear_thresh: float | None = None):
        n_mixture = log_scale.shape[-1] # TODO: this is very bad
        
        
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
    def identity_params(x: Tensor) -> dict[str, Tensor]:
        return {"log_scale": torch.zeros_like(x), "shift": None, "weights": None}

    def forward(self, x: Tensor) -> Tensor:
        s, t = self.log_scale, self.shift
        y = mod_2pi(2 * torch.atan(torch.exp(-s) * torch.tan((x - π) / 2) + t) + π)

        if self._linear_thresh is None:
            return y

        m1 = x < self._linear_thresh
        m2 = (2 * π - x) < self._linear_thresh

        y[m1] = (x * torch.exp(s))[m1]
        y[m2] = 2 * π - ((2 * π - x) * torch.exp(s))[m2]


    def inverse(self, y: Tensor) -> Tensor:
        s, t = self.log_scale, self.shift
        return mod_2pi(2 * torch.atan(torch.exp(s) * torch.tan(y / 2) - t))


    def _log_det_jacobian_single(self, x: Tensor, y: Tensor) -> Tensor:
        s, t = self.log_scale, self.shift
        dydx_recip = (
                (1 + t ** 2) * torch.exp(s) * torch.sin(x / 2) ** 2
                + torch.exp(-s) * torch.cos(x / 2) ** 2
                - t * torch.sin(x)
        )
        return dydx_recip.log().negative()

