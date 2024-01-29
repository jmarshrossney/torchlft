from math import pi as π
from typing import Any, Callable, TypeAlias

import torch

#from auxflows.utils import mod_2pi

Tensor: TypeAlias = torch.Tensor
Transform: TypeAlias = Callable[[Tensor, Any, ...], tuple[Tensor, Tensor]]
Wrapper: TypeAlias = Callable[Transform, Transform]


def complex_to_arg(transform):
    def wrapped_transform(x: Tensor, *args, **kwargs):
        r, θ = x.abs(), mod_2pi(x.angle())
        # NOTE: should I demand r == 1 here to be safe??
        ψ, grad_or_ldj = transform(θ, *args, **kwargs)
        y = torch.polar(r, ψ)
        return y, grad_or_ldj

    return wrapped_transform


def unit_vector_to_angle(transform):
    def wrapped_transform(x: Tensor, *args, **kwargs):
        raise NotImplementedError

    return wrapped_transform


def complex_to_vector(transform):
    def wrapped_transform(x: Tensor, *args, **kwargs):
        assert x.shape[-1] == 1
        x_2d = torch.cat([x.real, x.imag], dim=-1)
        y_2d, grad_or_ldj = transform(x_2d, *args, **kwargs)
        re_y, im_y = y_2d.split(1, dim=-1)
        y = torch.complex(re_y, im_y)
        return y, grad_or_ldj

    return wrapped_transform


def complex_to_polar(transform):
    def wrapped_transform(x: Tensor, *args, **kwargs):
        assert x.shape[-1] == 1
        rθ = torch.cat([x.abs(), mod_2pi(x.angle())], dim=-1)
        ρφ, grad_or_ldj = transform(rθ)
        ρ, φ = ρφ.split(1, dim=-1)
        y = torch.polar(ρ, φ)

        # Assume that r -> ρ went first, then θ -> φ
        ldj = grad_or_ldj
        ldj = ldj + ρ.log().flatten(1).sum(1, keepdim=True)

        return y, ldj

    return wrapped_transform


def vector_to_polar(transform):
    def wrapped_transform(x: Tensor, *args, **kwargs):
        raise NotImplementedError

    return wrapped_transform


def pi_rotation(transform):
    def wrapped_transform(θ: Tensor, *args, **kwargs):
        θ, grad_or_ldj = transform(mod_2pi(θ + π))
        return mod_2pi(θ - π), grad_or_ldj

    return wrapped_transform


def mask_outside_interval(
    transform,
    *,
    lower_bound: float,
    upper_bound: float,
    tol: float,
):
    assert lower_bound < upper_bound

    def wrapped_transform(x: Tensor, *args, **kwargs):
        inside_bounds = (x > lower_bound + tol) & (x < upper_bound - tol)

        y, grad = transform(
            x.clamp(lower_bound + tol, upper_bound - tol), *args, **kwargs
        )

        y = torch.where(inside_bounds, y, x)
        # NOTE: require grad, not ldj!
        grad = torch.where(inside_bounds, grad, torch.ones_like(grad))

        return y, grad

    return wrapped_transform


def sum_log_gradient(transform):
    def wrapped_transform(x: Tensor, *args, **kwargs):
        y, grad = transform(x, *args, **kwargs)
        sum_log_grad = grad.log().flatten(start_dim=1).sum(dim=1, keepdim=True)
        return y, sum_log_grad

    return wrapped_transform
