from collections import OrderedDict
from collections.abc import Iterator
from copy import deepcopy
import logging
import os
import pathlib
from typing import Union

import torch
from jsonargparse.typing import PositiveInt, NonNegativeInt, OpenUnitInterval
import tqdm

log = logging.getLogger(__name__)

__all__ = [
    "Sampler",
    "OneShotSampler",
]


class Sampler(torch.nn.Module):
    def __init__(self, **hparams):
        super().__init__()

        self.__dict__.update(hparams)

        self._global_step = 0
        self._transitions = 0
        self._state = self.init()

    @property
    def global_step(self) -> NonNegativeInt:
        return self._global_step

    @property
    def transitions(self) -> NonNegativeInt:
        return self._transitions

    @property
    def transition_rate(self) -> OpenUnitInterval:
        return (
            self._transitions / self._global_step if self._global_step else 0
        )

    @property
    def sweep_length(self) -> PositiveInt:
        return 1

    def get_state(self) -> dict:
        return deepcopy(self._state)

    def set_extra_state(self, state: dict) -> None:
        assert isinstance(state, dict), f"expected dict, but got {type(state)}"
        self._global_step = state["global_step"]
        torch.random.set_rng_state(state["rng_state"])
        self._state = state["state"]

    def get_extra_state(self) -> dict:
        return {
            "global_step": self.global_step,
            "rng_state": torch.random.get_rng_state(),
            "state": self.state,
        }

    @staticmethod
    def metropolis_test(delta_log_weight: torch.Tensor) -> bool:
        return delta_log_weight > 0 or delta_log_weight.exp() > torch.rand(1)

    def forward(
        self, size: PositiveInt = 1, interval: PositiveInt = 1
    ) -> OrderedDict:

        output = OrderedDict()

        with tqdm.trange(
            size,
            desc="Sampling",
            # bar_format="{l_bar}{bar}|{n_fmt}{postfix}",
            # bar_format="{desc}: {n_fmt}/{total_fmt} {bar}",
            postfix={"step": f"{self._global_step}"},
        ) as pbar:
            for _ in pbar:
                for __ in range(interval):
                    for ___ in range(self.sweep_length):
                        self._global_step += 1
                        result = self.update(self._state)
                        self._transitions += int(bool(result))

                    pbar.set_postfix({"step": f"{self._global_step}"})

                output[self._global_step] = deepcopy(self._state)

        return output

    def init(self) -> dict:
        """
        Returns an initial state.
        """
        raise NotImplementedError

    def update(self, state: dict) -> bool:
        """
        Update the current state.
        """
        raise NotImplementedError


class OneShotSampler(Sampler):
    def __init__(
        self,
        generator: Iterator[torch.Tensor, torch.Tensor],
    ):
        self.generator = generator
        super().__init__()

    def init(self) -> dict:
        config, log_weight = next(self.generator)
        return {"config": config, "log_weight": log_weight}

    def update(self, state: dict) -> bool:
        config, log_weight = next(self.generator)

        delta_log_weight = log_weight - state["log_weight"]

        if self.metropolis_test(delta_log_weight):
            state.update(config=config, log_weight=log_weight)
            return True
        return False
