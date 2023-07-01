from functools import partial
from math import pi as π
from typing import Optional

import torch
import torch.nn.functional as F

from flows_on_spheres.nn import TransformModule, CircularTransformModule
from flows_on_spheres.transforms.mixture import (
    make_mixture,
    normalise_weights,
    invert_bisect,
)
from flows_on_spheres.transforms.typing import TransformFunc, Transform

Tensor = torch.Tensor

# Credit to https://arxiv.org/pdf/2110.00351.pdf
# and https://github.com/noegroup/bgflow for this transformation


def exponential_ramp(
    x: Tensor, params: Tensor, *, power: int = 2, eps: float = 1e-9
) -> tuple[Tensor, Tensor]:
    assert isinstance(power, int) and power > 0
    a, b, ε = params, power, eps
    x_masked = torch.where(x > ε, x, torch.full_like(x, ε))
    exp_factor = -a * x_masked.pow(-b)
    ρ = torch.where(
        x > ε,
        torch.exp(exp_factor) / torch.exp(-a),
        torch.zeros_like(x),
    )
    dρdx = (-b / x_masked) * exp_factor * ρ
    return ρ, dρdx


def monomial_ramp(x: Tensor, params: Tensor, *, power: int = 2) -> Tensor:
    assert isinstance(power, int) and power > 0
    z = x ** (power - 1)
    return x * z, power * z


def _sigmoid(ramp: TransformFunc) -> TransformFunc:
    def sigmoid(x: Tensor, params: Tensor, **kwargs):
        ρ_x, dρdx_x = ramp(x, params, **kwargs)
        ρ_1mx, dρdx_1mx = ramp(1 - x, params, **kwargs)

        σ = ρ_x / (ρ_x + ρ_1mx)

        dσdx = (ρ_1mx * dρdx_x + ρ_x * dρdx_1mx) / (ρ_x + ρ_1mx) ** 2

        return σ, dσdx

    return sigmoid


def _affine_sigmoid(
    sigmoid: TransformFunc,
) -> TransformFunc:
    def affine_sigmoid(x: Tensor, params: Tensor, **kwargs):
        sigmoid_params, α, β = params.tensor_split([-2, -1], dim=-1)
        σ, dσdx = sigmoid((x - β) * α + 0.5, sigmoid_params, **kwargs)
        return σ, α * dσdx

    return affine_sigmoid


def _mix_with_identity(transform: TransformFunc) -> TransformFunc:
    def identity_mixture(
        x: Tensor, params: Tensor, **kwargs
    ) -> tuple[Tensor, Tensor]:
        params, c = params.tensor_split([-1], dim=-1)
        # assert torch.all((c >= 0) and (c <= 1))

        y, dydx = transform(x, params, **kwargs)

        y = c * y + (1 - c) * x
        dydx = c * dydx + (1 - c)

        return y, dydx

    return identity_mixture


def _rescale_to_interval(
    transform: TransformFunc,
    lower_bound: float,
    upper_bound: float,
) -> TransformFunc:
    lo, up = lower_bound, upper_bound
    assert lo < up

    def rescaled_transform(x: Tensor, *args, **kwargs):
        y, dydx = transform((x - lo) / (up - lo), *args, **kwargs)
        return y * (up - lo) + lo, dydx

    return rescaled_transform


def _forward_transform(
    ramp: str,
    n_mixture: int,
    weighted: bool,
    lower_bound: float,
    upper_bound: float,
) -> TransformFunc:
    assert ramp in ("monomial", "exponential")
    ramp = monomial_ramp if ramp == "monomial" else exponential_ramp

    sigmoid = _sigmoid(ramp)
    affine_sigmoid = _affine_sigmoid(sigmoid)
    transform = _mix_with_identity(affine_sigmoid)
    rescaled_transform = _rescale_to_interval(
        transform, lower_bound, upper_bound
    )

    if n_mixture > 1:
        return make_mixture(
            rescaled_transform, weighted=weighted, mixture_dim=-2
        )
    else:
        return rescaled_transform


def sigmoid_transform(
    params_are_batched: bool,
    n_mixture: int,
    weighted: bool = True,
    circular: bool = False,
    ramp: str = "exponential",
    ramp_kwargs: Optional[dict] = None,
    min_weight: float = 1e-2,
    invert_bisect_tol: float = 1e-3,
    invert_bisect_max_iter: int = 100,
) -> Transform:
    weighted = weighted if n_mixture > 1 else False

    in_dims = (0, 0) if params_are_batched else (0, None)
    vmap = partial(torch.vmap, in_dims=in_dims, out_dims=(0, 0))

    forward_fn = vmap(
        _forward_transform(
            ramp,
            n_mixture=n_mixture,
            weighted=weighted,
            lower_bound=0 if circular else -1,
            upper_bound=2 * π if circular else +1,
        )
    )
    inverse_fn = invert_bisect(
        forward_fn,
        0 if circular else -1,
        2 * π if circular else +1,
        tol=invert_bisect_tol,
        max_iter=invert_bisect_max_iter,
    )

    funcs = [lambda x: x.negative().exp(), torch.sigmoid, torch.sigmoid]
    if ramp == "exponential":
        funcs.insert(0, lambda x: F.softplus(x) + 1e-3)
    if weighted:
        funcs.append(partial(normalise_weights, dim=-2, min=min_weight))

    def handle_params(params: Tensor) -> Tensor:
        params = params.unflatten(-1, (n_mixture, -1)).split(1, dim=-1)
        return torch.cat(
            [func(param) for func, param in zip(funcs, params, strict=True)],
            dim=-1,
        ).squeeze(dim=-2)

    class SigmoidTransform:
        def __init__(self, params: Tensor):
            self.params = handle_params(params)

        def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
            return forward_fn(x, self.params)

        def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
            return inverse_fn(y, self.params)

    return SigmoidTransform


class SigmoidModule(TransformModule):
    def __init__(
        self,
        *,
        n_mixture: int,
        net_hidden_shape: Optional[list[int]] = None,
        net_activation: Optional[str] = None,
        weighted: bool = True,
        min_weight: float = 1e-2,
        ramp: str = "exponential",
        ramp_kwargs: Optional[dict] = None,
        invert_bisect_tol: float = 1e-3,
        invert_bisect_max_iter: int = 100,
    ):
        super().__init__(
            n_params=(4 + int(weighted)) * n_mixture,
            net_hidden_shape=net_hidden_shape,
            net_activation=net_activation,
        )
        self._transform = sigmoid_transform(
            params_are_batched=net_hidden_shape is not None,
            n_mixture=n_mixture,
            weighted=weighted,
            circular=False,
            ramp=ramp,
            ramp_kwargs=ramp_kwargs,
            min_weight=min_weight,
            invert_bisect_tol=invert_bisect_tol,
            invert_bisect_max_iter=invert_bisect_max_iter,
        )

    def transform(self, params: Tensor) -> Transform:
        return self._transform(params)


class CircularSigmoidModule(CircularTransformModule):
    def __init__(
        self,
        *,
        n_mixture: int,
        weighted: bool = True,
        min_weight: float = 1e-2,
        ramp: str = "exponential",
        ramp_kwargs: Optional[dict] = None,
        invert_bisect_tol: float = 1e-3,
        invert_bisect_max_iter: int = 100,
        net_hidden_shape: Optional[list[int]] = None,
        net_activation: Optional[str] = None,
    ):
        weighted = weighted if n_mixture > 1 else False
        super().__init__(
            n_params=(4 + int(weighted)) * n_mixture,
            net_hidden_shape=net_hidden_shape,
            net_activation=net_activation,
        )
        self._transform = sigmoid_transform(
            params_are_batched=net_hidden_shape is not None,
            n_mixture=n_mixture,
            weighted=weighted,
            circular=True,
            ramp=ramp,
            ramp_kwargs=ramp_kwargs,
            min_weight=min_weight,
            invert_bisect_tol=invert_bisect_tol,
            invert_bisect_max_iter=invert_bisect_max_iter,
        )

    def transform(self, params: Tensor) -> Transform:
        return self._transform(params)
