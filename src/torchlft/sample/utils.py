from collections.abc import Iterable
import itertools
import math
from typing import Union

from jsonargparse.typing import PositiveInt
import torch


def metropolis_test(delta_log_weight: Union[float, torch.Tensor]) -> bool:
    return delta_log_weight > 0 or math.exp(delta_log_weight) > torch.rand(1)


def autocorrelation(
    observable: Union[Iterable[float], torch.Tensor]
) -> torch.Tensor:
    if not isinstance(observable, torch.Tensor):
        observable = torch.tensor(observable)
    assert observable.dim() == 1
    n = observable.shape[0]
    observable = observable.view(1, 1, -1)

    autocovariance = torch.nn.functional.conv1d(
        torch.nn.functional.pad(observable, (0, n - 1)),
        observable,
    ).squeeze()

    return autocovariance.div(autocovariance[0])


def build_neighbour_list(
    lattice_shape: Iterable[PositiveInt],
) -> list[list[PositiveInt]]:
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
