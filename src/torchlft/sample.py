from __future__ import annotations

import os
import pathlib
from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.utils.tensorboard as tensorboard
import tqdm

if TYPE_CHECKING:
    from torchlft.typing import *


class SamplingAlgorithm(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self._global_step = None

        # Initialise empty buffer for state, transitions
        # NOTE: possibly better to force cloning of state?
        # TODO: figure out approach for states that are tuples of tensors
        self.register_buffer("state", None)
        self.register_buffer("_transitions", None)

        self.requires_grad_(False)
        self.train(False)

        self.init = MethodType(self._init_wrapper(self.init), self)
        self.register_forward_pre_hook(self._forward_pre_hook)
        self.register_forward_hook(self._forward_post_hook)

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def transitions(self) -> LongTensor:
        return self._transitions

    @property
    def context(self) -> dict:
        return {}

    @property
    def pbar_stats(self) -> dict:
        return {"steps": self._global_step}

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
        transitions = (
            output.long()
            if isinstance(output, torch.Tensor)
            else (output if type(output) is not None else 1)
        )
        self_._transitions += transitions

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


def metropolis_test(current: Tensor, proposal: Tensor) -> BoolTensor:
    # NOTE: torch.exp(large number) returns 'inf' and
    # ('inf' > x) for float x returns True, as required
    return (proposal - current).exp() > torch.rand_like(current)


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

    @torch.no_grad()
    def forward(self) -> bool:
        state, log_weight = next(self.generator)

        accepted = metropolis_test(
            current=self.log_weight, proposal=log_weight
        )

        self.state[accepted] = state[accepted]
        self.log_weight[accepted] = log_weight[accepted]

        return accepted


class HybridMonteCarlo(SamplingAlgorithm):
    def __init__(
        self,
        hamiltonian: Hamiltonian,
        integrator: Callable[
            [Tensor, Tensor, Hamiltonian], [Tensor, Tensor, float]
        ],
    ):
        super().__init__()
        self.hamiltonian = hamiltonian
        self.integrator = integrator

        self._delta_hamiltonian = []

    def exp_delta_hamiltonian(self) -> Tensor:
        return torch.stack(self._delta_hamiltonian).exp().mean(dim=0)

    def forward(self) -> BoolTensor:
        x0 = self.state
        p0 = self.hamiltonian.sample_momenta(x0)
        H0 = self.hamiltonian.compute(x0, p0)

        xT, pT, T = self.integrator(
            coords=x0,
            momenta=p0,
            hamiltonian=self.hamiltonian,
            step_size=self.step_size,
            traj_length=self.traj_length,
        )

        HT = self.hamiltonian.compute(xT, pT)

        self._delta_hamiltonian = H0 - HT

        accepted = metropolis_test(current=-H0, proposal=-HT)

        self.state[accepted] = xT[accepted]

        return accepted


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
    def logger(self) -> tensorboard.writer.SummaryWriter | None:
        return getattr(self, "_logger", None)

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
    ) -> list[Tensor | tuple[Tensor, ...]]:
        output = []
        with tqdm.trange(
            size, desc="Sampling", postfix=self.algorithm.pbar_stats
        ) as pbar:
            for i in pbar:
                self._sample(interval)
                output.append(self.algorithm.state)
                pbar.set_postfix(self.algorithm.pbar_stats)

        self._algorithm.on_final_sample()

        # NOTE: how important is it to close the logger?
        if hasattr(self, "_logger"):
            self._logger.flush()

        return configs
