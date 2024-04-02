from functools import partial
from math import pi as π
from typing import Optional, TypeAlias

import torch

from flows_on_spheres.nn import CircularTransformModule
from flows_on_spheres.geometry import as_angle, as_vector
from flows_on_spheres.linalg import dot_keepdim
from flows_on_spheres.utils import mod_2pi
from flows_on_spheres.transforms.mixture import (
    make_mixture,
    normalise_weights,
    invert_bisect,
)
from flows_on_spheres.transforms.typing import Transform

Tensor: TypeAlias = torch.Tensor


def _forward_transform(x: Tensor, ω: Tensor) -> tuple[Tensor, Tensor]:
    x_x0 = torch.stack([as_vector(x), as_vector(torch.zeros_like(x))])

    dydx = (1 - dot_keepdim(ω, ω)) / dot_keepdim(x_x0 - ω, x_x0 - ω)
    y_y0 = dydx * (x_x0 - ω) - ω

    y, y0 = as_angle(y_y0)
    y = mod_2pi(y - y0)

    dydx, _ = dydx

    return y, dydx


def _inverse_transform(y: Tensor, ω: Tensor) -> tuple[Tensor, Tensor]:
    y_y0 = torch.stack([as_vector(y), as_vector(torch.zeros_like(y))])

    dxdy = (1 - dot_keepdim(ω, ω)) / dot_keepdim(y_y0 + ω, y_y0 + ω)
    x_x0 = dxdy * (y_y0 + ω) + ω

    x, x0 = as_angle(x_x0)
    x = mod_2pi(x - x0)

    dxdy, _ = dxdy

    return x, dxdy


def to_unit_disk(ω: Tensor, ε: float) -> Tensor:
    assert ω.shape[-1] == 2
    ω = torch.tanh(ω) * (1 - ε)
    ω1, ω2 = ω.split(1, dim=-1)
    return torch.cat([ω1, ω2 * (1 - ω1**2).sqrt()], dim=-1)


def mobius_transform(
    params_are_batched: bool,
    n_mixture: int,
    weighted: bool = True,
    min_weight: float = 1e-2,
    bounds_tol: float = 1e-2,
    invert_bisect_tol: float = 1e-3,
    invert_bisect_max_iter: int = 100,
) -> Transform:
    weighted = weighted if n_mixture > 1 else False

    in_dims = (0, 0) if params_are_batched else (0, None)
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

    funcs = [partial(to_unit_disk, ε=bounds_tol)]
    if weighted:
        funcs.append(partial(normalise_weights, dim=-2, min=min_weight))

    def handle_params(params: Tensor) -> Tensor:
        params = params.unflatten(-1, (n_mixture, -1)).tensor_split(
            [2], dim=-1
        )
        # NOTE zip with strict=False so empty tensor dropped when unweighted
        return torch.cat(
            [func(param) for func, param in zip(funcs, params, strict=False)],
            dim=-1,
        ).squeeze(dim=-2)

    class MobiusTransform:
        def __init__(self, params: Tensor):
            self.params = handle_params(params)

        def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
            return forward_fn(x, self.params)

        def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
            return inverse_fn(y, self.params)

    return MobiusTransform


class MobiusModule(CircularTransformModule):
    def __init__(
        self,
        n_mixture: int,
        weighted: bool = True,
        min_weight: float = 1e-2,
        bounds_tol: float = 1e-2,
        invert_bisect_tol: float = 1e-3,
        invert_bisect_max_iter: int = 100,
        net_hidden_shape: Optional[list[int]] = None,
        net_activation: Optional[str] = None,
    ):
        super().__init__(
            n_params=(2 + int(weighted)) * n_mixture,
            net_hidden_shape=net_hidden_shape,
            net_activation=net_activation,
        )
        self._transform = mobius_transform(
            params_are_batched=net_hidden_shape is not None,
            n_mixture=n_mixture,
            weighted=weighted,
            min_weight=min_weight,
            bounds_tol=bounds_tol,
            invert_bisect_tol=invert_bisect_tol,
            invert_bisect_max_iter=invert_bisect_max_iter,
        )

    def transform(self, params: Tensor) -> Transform:
        return self._transform(params)
