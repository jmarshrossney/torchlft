from collections import OrderedDict
from collections.abc import Collection, Iterator, Iterable
from copy import deepcopy
import logging
import math
import os
import pathlib
from typing import Union

import torch
from jsonargparse.typing import PositiveInt, NonNegativeInt, OpenUnitInterval
import tqdm

from torchlft.sample.utils import metropolis_test, random_site_generator

log = logging.getLogger(__name__)

__all__ = [
    "SamplingAlgorithm",
    "RandomWalkMetropolis",
    "MCMCReweighting",
]


class SamplingAlgorithm(torch.nn.Module):
    def __init__(self, **hparams):
        super().__init__()

        self.__dict__.update(hparams)

        self._global_step = 0
        self._transitions = 0

        # Initialise empty buffer
        self.register_buffer("state", torch.tensor(0))

        self.register_forward_pre_hook(self.forward_pre_hook)
        self.register_forward_hook(self.forward_post_hook)

        # Run user-defined init
        self.init()

        self.requires_grad_(False)
        self.train(False)

    @property
    def global_step(self) -> NonNegativeInt:
        return self._global_step

    @property
    def transitions(self) -> NonNegativeInt:
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
    def forward_pre_hook(model, input: None) -> None:
        model._global_step += 1

    @staticmethod
    def forward_post_hook(
        model, input: None, output: Union[bool, None]
    ) -> None:
        model._transitions += int(output) if type(output) is not None else 1

    def init(self) -> None:
        """
        Initialises the sampler.
        """
        raise NotImplementedError

    def forward(self) -> Union[bool, None]:
        """
        Executes one step.
        """
        raise NotImplementedError


class RandomWalkMetropolis(SamplingAlgorithm):
    def __init__(
        self, lattice_shape: Iterable[PositiveInt], **hparams
    ) -> None:
        super().__init__(lattice_shape=lattice_shape, **hparams)
        self.random_site_generator = random_site_generator(lattice_shape)

    @property
    def sweep_length(self) -> PositiveInt:
        return math.prod(self.lattice_shape)

    def update(self, site: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self) -> Union[bool, None]:
        site_idx, neighbour_idx = next(self.random_site_generator)

        site, *neighbours = self.state.view(-1)[[site_idx, *neighbour_idx]]

        curr_local_action = self.local_action(site, neighbours)

        site = self.update(site)

        new_local_action = self.local_action(site, neighbours)

        delta_log_weight = new_local_action - curr_local_action
        if metropolis_test(delta_log_weight):
            self.state.view(-1)[site_idx] = site
            return True
        else:
            return False


class MCMCReweighting(SamplingAlgorithm):
    def __init__(
        self,
        generator: Iterator[torch.Tensor, torch.Tensor],
    ):
        self.generator = generator
        super().__init__()

    @property
    def context(self) -> dict:
        return {"log_weight": self.log_weight}

    def init(self) -> None:
        state, log_weight = next(self.generator)
        self.state = state
        self.log_weight = log_weight

    def forward(self) -> Union[bool, None]:
        state, log_weight = next(self.generator)

        delta_log_weight = log_weight - self.log_weight

        if metropolis_test(delta_log_weight):
            self.state = state
            self.log_weight = log_weight
            return True
        else:
            return False
