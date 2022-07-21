import itertools

import torch


def make_checkerboard(lattice_shape: list[int]) -> torch.BoolTensor:
    """
    Returns a boolean mask that selects 'even' lattice sites.
    """
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


def alternating_checkerboard_mask(lattice_shape: list[int]) -> itertools.cycle:
    """
    Infinite iterator which alternates between even and odd sites of the mask.
    """
    checker = make_checkerboard(lattice_shape)
    return itertools.cycle([checker, ~checker])


def laplacian_2d(lattice_length: int) -> torch.Tensor:
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


def nearest_neighbour_kernel(lattice_dim) -> torch.Tensor:
    identity_kernel = torch.zeros([3 for _ in range(lattice_dim)])
    identity_kernel.view(-1)[pow(3, lattice_dim) // 2] = 1

    nn_kernel = torch.zeros([3 for _ in range(lattice_dim)])
    for shift, dim in itertools.product([+1, -1], range(lattice_dim)):
        nn_kernel.add_(identity_kernel.roll(shift, dim))

    return nn_kernel
