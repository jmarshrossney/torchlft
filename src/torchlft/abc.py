from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from typing import TYPE_CHECKING

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
