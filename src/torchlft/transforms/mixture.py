from functools import partial, lru_cache
from typing import TypeAlias

import torch
import torch.nn.functional as F

from flows_on_spheres.transforms.typing import TransformFunc

Tensor: TypeAlias = torch.Tensor


class DidNotConvergeError(Exception):
    pass


def normalise_weights(weights: Tensor, dim: int, min: float = 1e-2) -> Tensor:
    d, ε = dim, min
    n_mix = weights.shape[d]
    assert n_mix * ε < 1
    return F.softmax(weights, dim=d) * (1 - n_mix * ε) + ε


@lru_cache
def make_mixture(
    transform: TransformFunc,
    weighted: bool = True,
    mixture_dim: int = -2,
) -> TransformFunc:
    d = mixture_dim
    vmapped_transform = torch.vmap(transform, (None, d), (d, d))

    if weighted:

        def mixture_transform(
            x: Tensor, params: Tensor, **kwargs
        ) -> tuple[Tensor, Tensor]:
            params, weights = params.tensor_split([-1], dim=-1)
            # NOTE: vmap/allclose support github.com/pytorch/functorch/issues/275
            #assert torch.allclose(weights.sum(dim=d), torch.ones(1))
            y, dydx = vmapped_transform(x, params, **kwargs)
            assert y.shape == weights.shape
            return (weights * y).sum(dim=d), (weights * dydx).sum(dim=d)

    else:

        def mixture_transform(
            x: Tensor, params: Tensor, **kwargs
        ) -> tuple[Tensor, Tensor]:
            y, dydx = vmapped_transform(x, params, **kwargs)
            return y.mean(dim=d), dydx.mean(dim=d)

    return mixture_transform


def invert_bisect(
    mixture_transform: TransformFunc,
    lower_bound: float,
    upper_bound: float,
    tol: float,
    max_iter: int,
):
    @torch.no_grad()
    def inverted_transform(y: Tensor, params: Tensor, **kwargs):
        f = partial(mixture_transform, **kwargs)

        x_low = torch.full_like(y, lower_bound)
        x_upp = torch.full_like(y, upper_bound)

        for _ in range(max_iter):
            x_trial = 0.5 * (x_low + x_upp)
            y_trial, dydx = f(x_trial, params)

            if (y - y_trial).abs().max() < tol:
                return x_trial, 1 / dydx

            if torch.all((x_trial == x_low) & (x_trial == x_upp)):
                raise ArithmeticError  # TODO

            x_low = torch.where(y > y_trial, x_trial, x_low)
            x_upp = torch.where(y > y_trial, x_upp, x_trial)

        raise DidNotConvergeError(
            "Failed to converge within specified n iters"
        )

    return inverted_transform
