from typing import TypeAlias

import torch

BoolTensor: TypeAlias = torch.BoolTensor


def make_2d_striped_mask(
    lattice: tuple[int, int], dim: int, period: int = 2, offset: int = 0
) -> BoolTensor:
    assert dim in Dimensions2d
    assert period >= 2

    μ, ν = Dimensions2d

    k = 1 if dim == ν else 2

    # NOTE: did I put the batch dim in for any reason?
    mask = (
        torch.zeros((1, *lattice), dtype=torch.bool)
        .unflatten(k, (-1, period))
        .index_fill(k + 1, torch.tensor(offset % period), True)
        .flatten(k, k + 1)
    )

    return mask.squeeze(0)


def checkerboard_mask(
    lattice: tuple[int, int],
    offset: int = 0,
    device: torch.device | None = None,
):
    # only works for even-valued lattice lengths
    assert lattice[0] % 2 == 0
    assert lattice[1] % 2 == 0

    checker = torch.zeros(lattice, dtype=torch.bool, device=device)
    checker[::2, ::2] = checker[1::2, 1::2] = True

    if offset % 2 == 1:
        checker = ~checker

    return checker


def dilated_checkerboard_mask(
    lattice: tuple[int, int],
    dilation: tuple[int, int],
    offset: tuple[int, int] = (0, 0),
):
    # only works for even-valued lattice lengths
    assert lattice[0] % 2 == 0
    assert lattice[1] % 2 == 0

    n, m = dilation
    δμ, δν = offset

    checker = torch.zeros(lattice, dtype=torch.bool)
    checker[:: (2 * n), :: (2 * m)] = True  # noqa: E203
    checker[n :: (2 * n), m :: (2 * m)] = True  # noqa: E203

    checker = checker.roll((-δμ, -δν), (0, 1))

    return checker
