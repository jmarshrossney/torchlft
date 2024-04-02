from functools import partial
from math import pi as π
from typing import Optional, TypeAlias

import torch

from flows_on_spheres.utils import mod_2pi
from flows_on_spheres.transforms.mixture import (
    make_mixture,
    normalise_weights,
    invert_bisect,
)
from torchlft.nflow.transforms.core import Transform

Tensor: TypeAlias = torch.Tensor


def _projection_forward(
    x: Tensor, params: Tensor, *, linear_thresh: float
) -> tuple[Tensor, Tensor]:
    α, β = params.split(1, dim=-1)
    ε = 0.01  # linear_thresh

    y = mod_2pi(2 * torch.atan(α * torch.tan((x - π) / 2) + β) + π)

    dxdy = (
        (1 + β**2) / α * torch.sin(x / 2) ** 2
        + α * torch.cos(x / 2) ** 2
        - β * torch.sin(x)
    )

    y = torch.where(x > ε, y, x / α)
    y = torch.where((2 * π - x) > ε, y, 2 * π - (2 * π - x) / α)
    dxdy = torch.where((x > ε) & ((2 * π - x) > ε), dxdy, α)

    dydx = 1 / dxdy

    return y, dydx


def _projection_inverse(
    y: Tensor, params: Tensor, *, linear_thresh: float
) -> tuple[Tensor, Tensor]:
    α, β = params.split(1, dim=-1)
    ε = 0.01  # linear_thresh

    x = mod_2pi(2 * torch.atan((1 / α) * torch.tan((y - π) / 2) - (β / α)) + π)

    dxdy = (
        (1 + β**2) / α * torch.sin(x / 2) ** 2
        + α * torch.cos(x / 2) ** 2
        - β * torch.sin(x)
    )

    x = torch.where(y > ε, x, y * α)
    x = torch.where((2 * π - y) > ε, x, 2 * π - (2 * π - y) * α)
    dxdy = torch.where((x > ε) & ((2 * π - x) > ε), dxdy, α)

    return x, dxdy


# NOTE: in future allow choice of inversion method, specified with
# inversion_strategy and inversion_kwargs
def projected_affine_transform(
    n_mixture: int,
    weighted: bool,
    batch_dims: int = 2,
    min_weight: float = 1e-2,
    linear_thresh: float = 1e-3,
    invert_bisect_tol: float = 1e-3,
    invert_bisect_max_iter: int = 100,
) -> Transform:

    vmap = partial(torch.vmap, in_dims=in_dims, out_dims=(0, 0))

    if n_mixture > 1:
        forward_fn = vmap(
            make_mixture(_forward_transform, weighted=weighted, mixture_dim=-2)
        )
        inverse_fn = invert_bisect(
            forward_fn,
            0,
            2 * π,
            tol=invert_bisect_tol,
            max_iter=invert_bisect_max_iter,
        )
    else:
        forward_fn = vmap(_forward_transform)
        inverse_fn = vmap(_inverse_transform)

    funcs = [lambda x: torch.exp(-x), lambda x: x]
    if weighted:
        funcs.append(partial(normalise_weights, dim=-2, min=min_weight))

    def handle_params(params: Tensor) -> Tensor:
        params = params.unflatten(-1, (n_mixture, -1)).split(1, dim=-1)
        return torch.cat(
            [func(param) for func, param in zip(funcs, params, strict=True)],
            dim=-1,
        ).squeeze(dim=-2)

    class ProjectedAffineTransform(Transform):
        n_params = ...
        handle_params_fn = staticmethod(handle_params)
        transform_fn = staticmethod(forward_fn)


    return ProjectedAffineTransform
