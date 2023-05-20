from __future__ import annotations

import os
import pathlib
from collections.abc import Iterator
from math import exp
from random import random
from typing import TYPE_CHECKING

import torch
import torch.utils.tensorboard as tensorboard
import tqdm

from torchlft.actions import (
    phi_four_action,
    phi_four_action_local,
)
from torchlft.utils.lattice import build_neighbour_list

if TYPE_CHECKING:
    from torchlft.typing import *


class SamplingAlgorithm(Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self._global_step = None
        self._transitions = None

        # Initialise empty buffer
        self.register_buffer("state", None)

        self.requires_grad_(False)
        self.train(False)

        self.init = MethodType(self._init_wrapper(self.init), self)
        self.register_forward_pre_hook(self._forward_pre_hook)
        self.register_forward_hook(self._forward_post_hook)

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def transitions(self) -> int:
        return self._transitions

    @property
    def context(self) -> dict:
        return {}

    @property
    def pbar_stats(self) -> dict:
        return {"steps": self._global_step, "moves": self._transitions}

    def set_extra_state(self, state: dict) -> None:
        assert isinstance(state, dict), f"expected dict, but got {type(state)}"
        self._global_step = state.pop("global_step")
        self._transitions = state.pop("transitions")
        torch.random.set_rng_state(state.pop("rng_state"))
        self.__dict__.update(state)

    def get_extra_state(self) -> dict:
        extra_context = dict(
            global_step=self._global_step,
            transitions=self._transitions,
            rng_state=torch.random.get_rng_state(),
        )
        return self.context | extra_context

    @staticmethod
    def _init_wrapper(init):
        @wraps(init)
        def wrapper(self):
            self._global_step = 0
            self._transitions = 0
            self.state = None
            init()

        return wrapper

    @staticmethod
    def _forward_pre_hook(self_, input: None) -> None:
        self_._global_step += 1

    @staticmethod
    def _forward_post_hook(self_, input: None, output: bool | None) -> None:
        self_._transitions += int(output) if type(output) is not None else 1

    @abstractmethod
    def init(self) -> None:
        ...

    @abstractmethod
    def forward(self) -> bool | None:
        ...

    def on_step(self) -> None:
        ...

    def on_sweep(self) -> None:
        ...

    def on_sample(self) -> None:
        ...

    def on_final_sample(self) -> None:
        ...


class Sampler:
    def __init__(
        self,
        algorithm: SamplingAlgorithm,
        output_dir: Optional[Union[str, os.PathLike]] = None,
    ):
        self._algorithm = algorithm
        if output_dir is not None:
            self._output_dir = pathlib.Path(str(output_dir)).resolve()
        else:
            self._output_dir = None
        self._run_idx = 0

        if hasattr(self._algorithm, "sweep_length"):
            self._update = self._sweep
        else:
            self._update = self._step

        self.init()

    @property
    def algorithm(self) -> SamplingAlgorithm:
        """
        Pointer to the sampling algorithm.
        """
        return self._algorithm

    @property
    def output_dir(self) -> pathlib.Path:
        """
        Directory for sampling outputs.
        """
        return self._output_dir

    @property
    def run_idx(self) -> int:
        return self._run_idx

    def _step(self) -> None:
        self._algorithm()
        self._algorithm.on_step()

    def _sweep(self) -> None:
        for _ in range(self._algorithm.sweep_length):
            self._step()
        self._algorithm.on_sweep()

    def _sample(self, interval: int) -> None:
        for _ in range(interval):
            self._update()
        self._algorithm.on_sample()

    def init(self) -> None:
        self._run_idx += 1
        if self._output_dir is not None:
            log_dir = str(self._output_dir / "logs" / f"run_{self._run_idx}")
            self._logger = tensorboard.writer.SummaryWriter(log_dir)
            self._algorithm.logger = self._logger
        self._algorithm.init()

    def thermalise(self, steps_or_sweeps: int) -> None:
        with tqdm.trange(
            steps_or_sweeps,
            desc="Thermalising",
            postfix=self._algorithm.pbar_stats,
        ) as pbar:
            for _ in pbar:
                self._update()
                pbar.set_postfix(self._algorithm.pbar_stats)

    def sample(
        self,
        size: int = 1,
        interval: int = 1,
    ):
        configs = torch.full_like(self._algorithm.state, float("NaN")).repeat(
            size, *(1 for _ in self._algorithm.state.shape)
        )
        with tqdm.trange(
            size, desc="Sampling", postfix=self._algorithm.pbar_stats
        ) as pbar:
            for i in pbar:
                self._sample(interval)
                configs[i] = self._algorithm.state
                pbar.set_postfix(self._algorithm.pbar_stats)

        self._algorithm.on_final_sample()

        # NOTE: how important is it to close the logger?
        if hasattr(self, "_logger"):
            self._logger.flush()

        return configs


def metropolis_test(delta_log_weight: float) -> bool:
    return delta_log_weight > 0 or exp(delta_log_weight) > random()


class MCMCReweighting(SamplingAlgorithm):
    def __init__(
        self,
        generator: Iterator[Tensor, Tensor],
    ):
        super().__init__()
        self.generator = generator

    @property
    def context(self) -> dict:
        return {"log_weight": self.log_weight}

    def init(self) -> None:
        state, log_weight = next(self.generator)
        self.state = state
        self.log_weight = log_weight

    def forward(self) -> bool:
        state, log_weight = next(self.generator)

        delta_log_weight = log_weight - self.log_weight

        if metropolis_test(delta_log_weight):
            self.state = state
            self.log_weight = log_weight
            return True
        else:
            return False


class RandomWalkMetropolis(SamplingAlgorithm):
    def __init__(
        self,
        lattice_shape: torch.Size,
        step_size: float,
        **couplings: dict[str, float],
    ) -> None:
        super().__init__()
        self.lattice_shape = lattice_shape
        self.step_size = step_size
        self.couplings = couplings

        self.lattice_size = math.prod(lattice_shape)
        self.neighbour_list = build_neighbour_list(lattice_shape)

    @property
    def sweep_length(self) -> int:
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
    r"""
    Hamiltonian (Hybrid) Monte Carlo sampling algorithm.

    Proposed updates are generated by introducing a set of fictitious Gaussian
    momenta :math:`\tilde\phi` and integrating the Hamiltonian system
    :math:`(\phi, \tilde\phi)` along an iso-density (fixed energy) contour.

    Momenta are drawn from a Gaussian

    .. math::

        \tilde\phi \sim \exp\left( \frac{1}{2} \tilde\phi^\top
        M^{-1} \tilde\phi \right)

    The integration is carried out using the leapfrog algorithm:

    .. math::

        \tilde\phi\left(t + \frac{\delta t}{2}\right) = \tilde\phi(t)
        - \frac{\delta t}{2} F(t)

        \phi(t + \delta t) = \phi(t) + \delta t M^{-1} \tilde\phi

        \tilde\phi(t+\delta t) = \tilde\phi\left(t+\frac{\delta t}{2}\right)
        - \frac{\delta t}{2} F(t+\delta t)

    At the end of the trajectory, the proposal is accepted or rejected based
    on the result of the Metropolis test

    Args:
        lattice_shape:
            Dimensions of the lattice
        trajectory_length:
            Total length of each molecular dynamics trajectory
        steps:
            Number of leapfrog updates to approximate the trajectory
        mass_matrix:
            A matrix of dimensions ``(lattice_size, lattice_size)`` that
            is the covariance matrix from which initial momenta are drawn.
            If not supplied, a unit diagonal covariance is used
        couplings:
            The couplings in the action
    """

    def __init__(
        self,
        lattice_shape: torch.Size,
        trajectory_length: int,
        steps: int,
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
        """
        Computes the generalised force driving the momentum update.

        The forces are computed by differentiating the action with respect to
        the generalised positions, i.e. the field configuration. In
        practice PyTorch's autograd machinery is used.
        """
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
