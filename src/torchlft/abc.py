from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from typing import TYPE_CHECKING, TypeAlias, Protocol

import torch


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


class BaseDensity(torch.nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def action(self, configs: Field) -> Tensor:
        ...

    @abstractmethod
    def sample(self, batch_size: int) -> Field:
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


class FieldTransform(torch.nn.Module, metaclass=ABCMeta):
    @property
    @abstractmethod
    def domain(self) -> Constraint:
        ...

    @property
    def codomain(self) -> Constraint:
        return self.domain

    @abstractmethod
    def forward(self, Φ: Field) -> tuple[Field, Tensor]:
        ...

    @abstractmethod
    def inverse(self, Ψ: Field) -> tuple[Field, Tensor]:
        ...
