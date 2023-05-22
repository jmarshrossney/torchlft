from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torchlft.typing import *


def assert_valid_partitioning(*masks: BoolTensor) -> None:
    union = torch.stack(masks).sum(dim=0)
    if (union == 1).all():
        return

    msg = ""
    if (union > 1).any():
        msg += "Some elements appear more than once\n"
    if (union == 0).any():
        msg += "Some elements do not appear\n"
    raise Exception(msg)  # TODO custom exc


def make_checkerboard(lattice_shape: list[int]) -> BoolTensor:
    """
    Returns a boolean mask that selects 'even' lattice sites.
    """
    assert all(
        [n % 2 == 0 for n in lattice_shape]
    ), "each lattice dimension should be even"

    # NOTE: interesting that torch.jit.trace fails for
    # torch.full(lattice_shape, False, device=device)
    checkerboard = torch.zeros(lattice_shape, dtype=torch.bool)

    if len(lattice_shape) == 1:
        checkerboard[::2] = True
    elif len(lattice_shape) == 2:
        checkerboard[::2, ::2] = True
        checkerboard[1::2, 1::2] = True
    else:
        raise NotImplementedError("d > 2 currently not supported")
    return checkerboard


def make_checkerboard_partitions(
    lattice_shape: torch.Size, device=torch.device
) -> list[BoolTensor, BoolTensor]:
    checker = make_checkerboard(lattice_shape, device)
    return [checker, ~checker]


def alternating_checkerboard_mask(lattice_shape: list[int]) -> itertools.cycle:
    """
    Infinite iterator which alternates between even and odd sites of the mask.
    """
    checker = make_checkerboard(lattice_shape)
    return itertools.cycle([checker, ~checker])


def laplacian_2d(lattice_length: int) -> Tensor:
    """
    Creates a 2d Laplacian matrix.

    This works by taking the kronecker product of the one-dimensional
    Laplacian matrix with the identity.

    For now assumes a square lattice. Periodic BCs are also assumed.
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


def nearest_neighbour_kernel(lattice_dim) -> Tensor:
    identity_kernel = torch.zeros([3 for _ in range(lattice_dim)])
    identity_kernel.view(-1)[pow(3, lattice_dim) // 2] = 1

    nn_kernel = torch.zeros([3 for _ in range(lattice_dim)])
    for shift, dim in itertools.product([+1, -1], range(lattice_dim)):
        nn_kernel.add_(identity_kernel.roll(shift, dim))

    return nn_kernel


def build_neighbour_list(
    lattice_shape: torch.Size,
) -> list[list[int]]:
    indices = torch.arange(math.prod(lattice_shape)).view(lattice_shape)
    lattice_dims = range(len(lattice_shape))
    neighbour_indices = torch.stack(
        [
            indices.roll(shift, dim).flatten()
            for shift, dim in itertools.product([1, -1], lattice_dims)
        ],
        dim=1,
    )
    return neighbour_indices.tolist()

def correlator_restore_geom(correlator_lexi: Tensor, lattice_shape: tuple[int, int]) -> Tensor:
    """
    Takes a volume average of 2-dimensional shifts
    For each lattice site (row), restores the geometry by representing
    the row as a 2d array where the axis correspond to displacements
    in the two lattice directions. Then takes a volume average by
    averaging over all rows (lattice sites)
    """
    L, T = lattice_shape
    return torch.stack(
        [
            row.view(L, T).roll(
                ((-i // T), (-i % L)),
                dims=(0, 1),
            )
            for i, row in enumerate(correlator_lexi)
        ],
        dim=0,
    ).mean(dim=0)
