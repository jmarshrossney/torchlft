from collections.abc import Iterable
import itertools
import math

from jsonargparse.typing import PositiveInt
import torch


def metropolis_test(delta_log_weight: torch.Tensor) -> bool:
    return delta_log_weight > 0 or delta_log_weight.exp() > torch.rand(1)


def autocorrelation(observable: torch.Tensor) -> torch.Tensor:
    assert observable.dim() == 2
    autocovariance = torch.nn.functional.conv1d(
        torch.nn.functional.pad(
            observable.unsqueeze(1), (0, len(observable) - 1)
        ),
        observable.unsqueeze(1),
    ).squeeze(1)
    autocorrelation = autocovariance.div(autocovariance[:, 0])
    return autocorrelation


def build_neighbour_list(lattice_shape: Iterable[PositiveInt]) -> torch.Tensor:
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


def random_site_generator(
    lattice_shape: Iterable[PositiveInt],
) -> tuple[PositiveInt, list[PositiveInt]]:
    neighbour_indices = build_neighbour_list(lattice_shape)
    lattice_size = math.prod(lattice_shape)

    while True:
        idx = torch.randint(0, lattice_size, [1]).item()
        neighbours = neighbour_indices[idx]
        yield idx, neighbours
