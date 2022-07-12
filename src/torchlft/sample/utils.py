from collections.abc import Iterable
import itertools
import math

from jsonargparse.typing import PositiveInt
import torch


def metropolis_test(delta_log_weight: torch.Tensor) -> bool:
    return delta_log_weight > 0 or math.exp(delta_log_weight) > torch.rand(1)


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


def build_neighbour_list(lattice_shape: Iterable[PositiveInt]) -> list[list[PositiveInt]]:
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
