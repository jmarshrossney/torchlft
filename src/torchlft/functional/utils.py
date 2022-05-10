from __future__ import annotations

import torch
import torch.nn.functional
from typing import Callable


def build_rational_quadratic_spline(
    widths: torch.Tensor,
    heights: torch.Tensor,
    derivs: torch.Tensor,
    interval: tuple[float, float] = (0, 1),
    pad_derivs_func: Callable[
        torch.Tensor, torch.Tensor
    ] = lambda derivs: torch.nn.functional.pad(derivs, (1, 1), "constant", 1),
) -> tuple[torch.Tensor]:
    r"""Builds a rational quadratic spline function.

    This uses the parametrisation introduced by [Gregory and Delbourgo]_

    Parameters
    ----------
    widths
        Un-normalised segment sizes in the domain
    heights
        Un-normalised segment sizes in the codomain
    derivs
        Unconstrained derivatives at the knots
    interval
        Tuple representing the lower and upper boundaries of the domain
        and the codomain, which are the same
    pad_derivs_func
        A function which adds a padding to the derivatives after they have
        been

    Returns
    -------
    widths
        Normalised segment sizes in the domain
    heights
        Normalised segment sizes in the codomain
    derivs
        Constrained derivatives at the knots
    knots_xcoords
        Coordinates of the knots in the domain
    knots_ycoords
        Coordinates of the knots in the codomain


    References
    ----------
    .. [Gregory and Delbourgo]
    Gregory, J. A. & Delbourgo, R. C2 Rational Quadratic Spline Interpolation
    to Monotonic Data, IMA Journal of Numerical Analysis, 1983, 3, 141-152
    """
    assert widths.shape == heights.shape
    assert derivs.shape[:-1] == heights.shape[:-1]
    """
    if endpoints == "linear":
        assert derivs.shape[-1] == heights.shape[-1] - 1
    elif endpoints == "circular":
        assert derivs.shape[-1] == heights.shape[-1]
    else:
        assert derivs.shape[-1] == heights.shape[-1] + 1
    """

    interval_size = interval[1] - interval[0]

    # Normalise the widths and heights to the interval
    widths = torch.nn.functional.softmax(heights, dim=-1).mul(interval_size)
    heights = torch.nn.functional.softmax(heights, dim=-1).mul(interval_size)

    # Let the derivatives be positive definite
    derivs = pad_derivs_func(torch.nn.functional.softplus(derivs))

    # Just a convenient way to ensure it's on the same device as heights
    zeros = torch.zeros_like(heights).sum(dim=-1, keepdim=True)

    knots_xcoords = (
        torch.cat(
            (
                zeros,
                torch.cumsum(widths, dim=-1),
            ),
            dim=-1,
        )
        + interval[0]
    )
    knots_ycoords = (
        torch.cat(
            (
                zeros,
                torch.cumsum(heights, dim=-1),
            ),
            dim=-1,
        )
        + interval[0]
    )
    return widths, heights, derivs, knots_xcoords, knots_ycoords
