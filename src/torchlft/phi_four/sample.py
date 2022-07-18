from collections.abc import Iterable
import math
from typing import Optional

from jsonargparse.typing import PositiveInt, PositiveFloat
import torch

from torchlft.phi_four.actions import (
    phi_four_action,
    phi_four_action_local,
)
from torchlft.sample.algorithms import SamplingAlgorithm
from torchlft.sample.utils import build_neighbour_list, metropolis_test


class RandomWalkMetropolis(SamplingAlgorithm):
    def __init__(
        self,
        lattice_shape: Iterable[PositiveInt],
        step_size: PositiveFloat,
        **couplings: dict[str, float],
    ) -> None:
        super().__init__()
        self.lattice_shape = lattice_shape
        self.step_size = step_size
        self.couplings = couplings

        self.lattice_size = math.prod(lattice_shape)
        self.neighbour_list = build_neighbour_list(lattice_shape)

    @property
    def sweep_length(self) -> PositiveInt:
        return self.lattice_size

    def init(self) -> None:
        self.state = torch.empty(self.lattice_shape).normal_(0, 1)

        # This is just a view of the original state
        self.flattened_state = self.state.view(-1)

    def forward(self) -> bool:
        site_idx = torch.randint(0, self.lattice_size, [1]).item()
        neighbour_idxs = self.neighbour_list[site_idx]

        phi_old, *neighbours = self.flattened_state[
            [site_idx, *neighbour_idxs]
        ]
        phi_new = phi_old + torch.randn(1).item() * self.step_size

        old_action = phi_four_action_local(
            phi_old, neighbours, **self.couplings
        )
        new_action = phi_four_action_local(
            phi_new, neighbours, **self.couplings
        )

        if metropolis_test(old_action - new_action):
            self.flattened_state[site_idx] = phi_new
            return True
        else:
            return False


class HamiltonianMonteCarlo(SamplingAlgorithm):
    def __init__(
        self,
        lattice_shape: Iterable[PositiveInt],
        trajectory_length: PositiveFloat,
        steps: PositiveInt,
        mass_matrix: Optional[torch.Tensor] = None,
        **couplings: dict[str, float],
    ) -> None:
        super().__init__()
        self.lattice_shape = lattice_shape
        self.trajectory_length = trajectory_length
        self.steps = steps
        self.couplings = couplings

        self.lattice_size = math.prod(lattice_shape)

        if mass_matrix is None:
            mass_matrix = torch.eye(self.lattice_size)
        self.momentum_distribution = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.lattice_size), covariance_matrix=mass_matrix
        )
        self.inverse_mass_matrix = self.momentum_distribution.precision_matrix

        # TODO: this is a bit hacky
        self.potential = lambda state: phi_four_action(
            state.view([1, *self.lattice_shape]), **self.couplings
        )

        # TODO: clean up namespace - too many attributes that might get
        # overridden by someone subclassing and logging things

    def init(self) -> None:
        self.state = torch.empty(self.lattice_shape).normal_(0, 1)

    def kinetic_term(self, momentum: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.dot(
            momentum, (torch.mv(self.inverse_mass_matrix, momentum))
        )

    def get_force(self, state: torch.Tensor) -> torch.Tensor:
        state.requires_grad_()
        state.grad = None
        with torch.enable_grad():
            self.potential(state).backward()
        force = state.grad
        state.requires_grad_(False)
        state.grad = None
        return force

    def forward(self) -> bool:

        state = self.state.clone().view(-1)
        momentum = self.momentum_distribution.sample()

        initial_hamiltonian = self.kinetic_term(momentum) + self.potential(
            state
        )

        delta = self.trajectory_length / self.steps

        # Begin leapfrog integration
        momentum -= delta / 2 * self.get_force(state)

        for _ in range(self.steps - 1):

            state = state.addmv(
                self.inverse_mass_matrix, momentum, alpha=delta
            )
            momentum -= delta * self.get_force(state)

        state.addmv_(self.inverse_mass_matrix, momentum, alpha=delta)
        momentum -= delta / 2 * self.get_force(state)

        final_hamiltonian = self.kinetic_term(momentum) + self.potential(state)

        if metropolis_test(initial_hamiltonian - final_hamiltonian):
            self.state = state.view(self.lattice_shape)
            return True
        else:
            return False
