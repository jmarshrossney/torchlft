from abc import ABCMeta, abstractmethod
from typing import NamedTuple, TypeAlias

import torch
import torch.nn as nn

from torchlft.utils.torch import tuple_concat

Tensor: TypeAlias = torch.Tensor

from math import exp
from random import random

def metropolis_hastings(log_weights: Tensor):
    assert log_weights.squeeze().dim() == 1

    log_weights = log_weights.squeeze().tolist()

    current = log_weights.pop(0)

    idx = 0
    indices = []

    for proposal in log_weights:
        if (proposal > current) or (
            random() < min(1, exp(proposal - current))
        ):
            current = proposal
            idx += 1

        indices.append(idx)

    indices = torch.tensor(indices, dtype=torch.long)

    return indices


class Model(nn.Module, metaclass=ABCMeta):
    class Fields(NamedTuple):
        inputs: tuple[Tensor, ...]
        outputs: tuple[Tensor, ...]

    class Actions(NamedTuple):
        base: Tensor
        target: Tensor
        pushforward: Tensor
        pullback: Tensor

    def __init__(self, lattice: tuple[int, ...], **couplings: float):
        super().__init__()
        self._lattice = lattice
        self._couplings = couplings
        self.register_buffer("_dummy_buffer", torch.tensor(0.0))

    @property
    def lattice(self) -> tuple[int, ...]:
        return self._lattice

    @lattice.setter
    def lattice(self, new: tuple[int, ...]) -> None:
        self._lattice = new

    @property
    def couplings(self) -> dict[str, float]:
        return self._couplings

    @property
    def device(self) -> torch.device:
        return self._dummy_buffer.device

    @property
    def dtype(self) -> torch.dtype:
        return self._dummy_buffer.dtype

    def forward(
        self, batch_size: int
    ) -> tuple["Model.Fields", "Model.Actions"]:
        inputs, base_action = self.base(batch_size)
        outputs, log_det_jacobian = self.flow_forward(inputs)
        target_action = self.target(outputs)

        pushforward = base_action + log_det_jacobian
        pullback = target_action - log_det_jacobian

        fields = self.Fields(inputs, outputs)
        actions = self.Actions(
            base_action, target_action, pushforward, pullback
        )

        return fields, actions

    @torch.no_grad()
    def weighted_sample(self, batch_size: int, n_batches: int = 1) -> tuple[Tensor, Tensor]:
        sample = []
        for _ in range(n_batches):
            fields, actions = self(batch_size)
            log_weights = actions.pushforward - actions.target
            sample.append([fields.outputs, log_weights])

        return tuple_concat(sample)

    @torch.no_grad()
    def metropolis_sample(self, batch_size: int, n_batches: int = 1) -> tuple[Tensor, Tensor]:
        outputs, log_weights = self.weighted_sample(batch_size, n_batches)
        indices = metropolis_hastings(log_weights)
        return (outputs, indices)


    @abstractmethod
    def flow_forward(
        self, inputs: tuple[Tensor, ...]
    ) -> tuple[tuple[Tensor, ...], Tensor]:
        ...

    @abstractmethod
    def base(
        self,
        batch_size: int,
    ) -> tuple[Tensor | tuple[Tensor, ...], Tensor]:
        ...

    @abstractmethod
    def target(
        self,
        inputs: tuple[Tensor, ...],
    ) -> Tensor:
        ...
