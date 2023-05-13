"""
Abstract base classes for the core components in torchlft.
"""

__all__ = [
    "BaseDensity",
    "Constraint",
    "Field",
    "ScalarField",
    "Transform",
]

from abc import ABC, ABCMeta, abstractmethod
import math

import torch

from torchlft.constraints import _Constraint, real
from torchlft.fields import _Field, _ScalarField
from torchlft.typing import *

Constraint: TypeAlias = _Constraint
Field: TypeAlias = _Field
ScalarField: TypeAlias = _ScalarField


class Action(ABC):
    ...


class CompositeField(ABC):
    ...  # TODO


class BaseDensity(Module, metaclass=ABCMeta):
    @abstractmethod
    def action(self, configs: Field) -> Tensor:
        ...

    @abstractmethod
    def sample(self, batch_size: int) -> Field:
        ...


class Transform(ABC):
    @property
    @abstractmethod
    def param_constraints(self) -> dict[str, Constraint]:
        ...

    @property
    @abstractmethod
    def domain(self) -> Constraint:
        ...

    @property
    @abstractmethod
    def codomain(self) -> Constraint:
        ...

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def inverse(self, y: Tensor) -> Tensor:
        ...

    @abstractmethod
    def log_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        ...

    def forward_and_ldj(self, x: Tensor) -> tuple[Tensor, Tensor]:
        y = self.forward(x)
        ldj = self.log_det_jacobian(x, y)
        return y, ldj

    def inverse_and_ldj(self, y: Tensor) -> tuple[Tensor, Tensor]:
        x = self.inverse(y)
        ldj = self.log_det_jacobian(x, y)
        return x, ldj
