from __future__ import annotations

import itertools
import math

import torch
import torch.nn.functional as F


def make_checkerboard(lattice_shape: list[int]) -> torch.BoolTensor:
    """Return a boolean mask that selects 'even' lattice sites."""
    assert all(
        [n % 2 == 0 for n in lattice_shape]
    ), "each lattice dimension should be even"
    checkerboard = torch.full(lattice_shape, False)
    if len(lattice_shape) == 1:
        checkerboard[::2] = True
    elif len(lattice_shape) == 2:
        checkerboard[::2, ::2] = True
        checkerboard[1::2, 1::2] = True
    else:
        raise NotImplementedError("d > 2 currently not supported")
    return checkerboard


def laplacian_2d(lattice_length: int) -> torch.Tensor:
    """Creates a 2d Laplacian matrix.

    This works by taking the kronecker product of the one-dimensional
    Laplacian matrix with the identity.

    Notes
    -----
    For now, assume a square lattice. Periodic BCs are also assumed.
    """
    identity = torch.eye(lattice_length)
    lapl_1d = (
        2 * identity  # main diagonal
        - torch.diag(torch.ones(lattice_length - 1), diagonal=1)  # upper
        - torch.diag(torch.ones(lattice_length - 1), diagonal=-1)  # lower
    )
    lapl_1d[0, -1] = lapl_1d[-1, 0] = -1  # periodicity
    lapl_2d = torch.kron(lapl_1d, identity) + torch.kron(identity, lapl_1d)
    return lapl_2d


def nearest_neighbour_kernel(lattice_dim) -> torch.Tensor:
    identity_kernel = torch.zeros([3 for _ in range(lattice_dim)])
    identity_kernel.view(-1)[pow(3, lattice_dim) // 2] = 1

    nn_kernel = torch.zeros([3 for _ in range(lattice_dim)])
    for shift, dim in itertools.product([+1, -1], range(lattice_dim)):
        nn_kernel.add_(identity_kernel.roll(shift, dim))

    return nn_kernel.view(1, 1, *nn_kernel.shape)


def build_rq_spline(
    widths: torch.Tensor,
    heights: torch.Tensor,
    derivs: torch.Tensor,
    interval: tuple[float, float],
    domain: str,
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
    domain
        One of 'interval', 'circle' and 'reals', which refers to the domain
        of the spline inputs. The number of derivatives provided is tied to
        this parameter #TODO document

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
    n_segments = widths.shape[-1]
    interval_size = interval[1] - interval[0]

    assert widths.shape == heights.shape
    assert derivs.shape[:-1] == heights.shape[:-1]
    if domain == "interval":
        assert derivs.shape[-1] == n_segments + 1
        pad_derivs = lambda derivs: derivs
    elif domain == "circle":
        assert derivs.shape[-1] == n_segments
        assert math.isclose(interval_size, 2 * math.pi)
        pad_derivs = lambda derivs: F.pad(derivs, (0, 1), "circular")
    elif domain == "reals":
        assert derivs.shape[-1] == n_segments - 1
        pad_derivs = lambda derivs: F.pad(derivs, (1, 1), "constant", 1)
    else:
        raise Exception(f"Invalid domain: '{domain}'")

    # Normalise the widths and heights to the interval
    widths = F.softmax(widths, dim=-1).mul(interval_size)
    heights = F.softmax(heights, dim=-1).mul(interval_size)

    # Let the derivatives be positive definite
    derivs = F.softplus(derivs)
    derivs = pad_derivs(derivs)

    # Just a convenient way to ensure it's on the correct device
    zeros = torch.zeros_like(widths).sum(dim=-1, keepdim=True)

    knots_xcoords = torch.cat(
        (
            zeros,
            torch.cumsum(widths, dim=-1),
        ),
        dim=-1,
    ).add(interval[0])
    knots_ycoords = torch.cat(
        (
            zeros,
            torch.cumsum(heights, dim=-1),
        ),
        dim=-1,
    ).add(interval[0])

    return widths, heights, derivs, knots_xcoords, knots_ycoords
