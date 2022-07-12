from collections import OrderedDict
from collections.abc import Collection, Iterator, Iterable
from copy import deepcopy
from functools import wraps
import logging
import math
from matplotlib.figure import Figure
from numbers import Real
import os
import pathlib
from types import MethodType
from typing import Union

import torch
from jsonargparse.typing import PositiveInt, NonNegativeInt, OpenUnitInterval
import tqdm

from torchlft.sample.utils import metropolis_test

log = logging.getLogger(__name__)

__all__ = [
    "SamplingAlgorithm",
    "MCMCReweighting",
]


class SamplingAlgorithm(torch.nn.Module):
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
    def _init_wrapper(init):
        @wraps(init)
        def wrapper(self):
            self._global_step = 0
            self._transitions = 0
            self.state = torch.tensor([])
            init()

        return wrapper

    @staticmethod
    def _forward_pre_hook(model, input: None) -> None:
        model._global_step += 1

    @staticmethod
    def _forward_post_hook(
        model, input: None, output: Union[bool, None]
    ) -> None:
        model._transitions += int(output) if type(output) is not None else 1

    def log(
        self,
        tag: str,
        value: Union[
            Real,
            str,
            dict[str, Real],
            list[Real],
            torch.Tensor,
            Figure,
        ],
    ) -> None:
        # TODO: handle exception when no logger defined
        # TODO: organise this better. Allow custom logging rules
        if not hasattr(self, "logger"):
            # TODO: warn user
            return

        if isinstance(value, Real):
            self.logger.add_scalar(tag, value, self._global_step)
        elif isinstance(value, str):
            self.logger.add_text(tag, value, self._global_step)
        elif isinstance(value, dict):
            self.logger.add_scalars(tag, value, self._global_step)
        elif isinstance(value, list):
            self.logger.add_histogram(
                tag, torch.tensor(value).flatten(), self._global_step
            )
        elif isinstance(value, torch.Tensor):
            if value.numel() > 1:
                self.logger.add_histogram(
                    tag, value.flatten(), self._global_step
                )
            else:
                self.logger.add_scalar(tag, value.item(), self._global_step)
        elif isinstance(value, Figure):
            self.logger.add_figure(tag, value, self._global_step)
        else:
            raise TypeError(
                f"No logging rule found for data type {type(value)}"
            )

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

    def on_step(self) -> None:
        ...

    def on_sweep(self) -> None:
        ...

    def on_sample(self) -> None:
        ...

    def on_final_sample(self) -> None:
        ...


class MCMCReweighting(SamplingAlgorithm):
    def __init__(
        self,
        generator: Iterator[torch.Tensor, torch.Tensor],
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

    def forward(self) -> Union[bool, None]:
        state, log_weight = next(self.generator)

        delta_log_weight = log_weight - self.log_weight

        if metropolis_test(delta_log_weight):
            self.state = state
            self.log_weight = log_weight
            return True
        else:
            return False
