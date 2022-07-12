from collections.abc import Iterable
import math

from jsonargparse.typing import PositiveInt, PositiveFloat
import torch

from torchlft.phi_four.actions import phi_four_action_local as local_action
from torchlft.sample.algorithms import SamplingAlgorithm
from torchlft.sample.utils import build_neighbour_list, metropolis_test


class RandomWalkMetropolis(SamplingAlgorithm):
    def __init__(
        self,
        lattice_shape: Iterable[PositiveInt],
        step_size: PositiveFloat,
        **couplings: dict[str, float],
    ) -> None:
        super().__init__(
            lattice_shape=lattice_shape,
            step_size=step_size,
            couplings=couplings,
        )
        self.lattice_size = math.prod(lattice_shape)
        self.neighbour_list = build_neighbour_list(lattice_shape)

        # This is just a view of the original state
        self.flattened_state = self.state.view(-1)

    @property
    def sweep_length(self) -> PositiveInt:
        return self.lattice_size

    def init(self) -> None:
        self.state = torch.empty(self.lattice_shape).normal_().flatten()

    def forward(self) -> bool:
        site_idx = torch.randint(0, self.lattice_size, [1]).item()
        neighbour_idxs = self.neighbour_list[site_idx]

        phi_old, *neighbours = self.flattened_state[
            [site_idx, *neighbour_idxs]
        ]
        phi_new = phi_old + torch.randn(1).item() * self.step_size

        old_action = local_action(phi_old, neighbours, **self.couplings)
        new_action = local_action(phi_new, neighbours, **self.couplings)

        if metropolis_test(new_action - old_action):
            self.flattened_state[site_idx] = phi_new
            return True
        else:
            return False


class HamiltonianMonteCarlo(SamplingAlgorithm):
    pass
