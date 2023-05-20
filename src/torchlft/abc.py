from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from typing import TYPE_CHECKING, TypeAlias, Protocol

import torch
import torch.nn as nn

from torchlft.fields import (
    Field,
    CanonicalField,
    PartitionedField,
    MaskedField,
    CompositeField,
    ScalarField,
)

if TYPE_CHECKING:
    from torchlft.constraints import Constraint

Tensor = torch.Tensor


# ABC or protocol?
class BaseAction(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def compute(self, configs: Tensor | tuple[Tensor, ...]) -> Tensor:
        ...

    @abstractmethod
    def gradient(self, configs: Tensor | tuple[Tensor, ...]) -> Tensor:
        ...

    @abstractmethod
    def sample(self, size: int) -> Tensor | tuple[Tensor, ...]:
        ...


class TargetAction(Protocol):
    def compute(self, configs: Tensor | tuple[Tensor, ...]) -> Tensor:
        ...

    def gradient(self, configs: Tensor | tuple[Tensor, ...]) -> Tensor:
        ...


class Geometry(Protocol):
    def partition(
        self, fields: Tensor | tuple[Tensor, ...]
    ) -> tuple[Tensor, ...] | tuple[tuple[Tensor, ...], ...]:
        ...

    def restore(
        self, partitions: tuple[Tensor, ...] | tuple[tuple[Tensor, ...], ...]
    ) -> Tensor | tuple[Tensor, ...]:
        ...
