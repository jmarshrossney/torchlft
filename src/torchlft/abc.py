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
from math import pi as π

import torch

from torchlft.constraints import _Constraint, real, periodic
from torchlft.typing import *

Constraint: TypeAlias = _Constraint

DEBUG = True


class Action(ABC):
    ...


class Field(ABC):
    def __init__(
        self,
        data,
        *,
        lattice_shape: torch.Size,
        element_shape: torch.Size,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:

        # as_tensor doesn't copy data -> not a leaf tensor
        # NOTE: this could be an issue?
        tensor = torch.as_tensor(data, dtype=dtype, device=device)

        if DEBUG:
            self.domain.check(tensor)

        self._tensor = tensor
        self._batch_size = tensor.shape[0]
        self._lattice_shape = tuple(lattice_shape)
        self._element_shape = tuple(element_shape)

    def __len__(self) -> int:
        return self._batch_size

    """
    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...
    """

    @property
    def lattice_shape(self) -> torch.Size:
        return self._lattice_shape

    @property
    def element_shape(self) -> torch.Size:
        return self._element_shape

    @property
    def tensor(self) -> Tensor:
        return self._tensor

    def clone(self) -> "Field":
        return self.new_like(self.tensor.clone())

    @abstractmethod
    def new_like(self, data: Tensor) -> "Field":
        ...

    @property
    @abstractmethod
    def domain(self) -> Constraint:
        ...

    @abstractmethod
    def to_canonical(self) -> "Field":
        ...

    @classmethod
    @abstractmethod
    def from_canonical(cls, other: "Field") -> "Field":
        ...


class ScalarField(Field):
    domain: Constraint = real

    def __pos__(self) -> "ScalarField":
        return self

    def __neg__(self) -> "ScalarField":
        return self.new_like(-self.tensor)

    def __add__(self, value: Tensor | float) -> "ScalarField":
        return self.new_like(self.tensor + value)

    def __sub__(self, value: Tensor | float) -> "ScalarField":
        return self.new_like(self.tensor - value)

    def __mul__(self, value: Tensor | float) -> "ScalarField":
        return self.new_like(self.tensor * value)

    def __div__(self, value: Tensor | float) -> "ScalarField":
        return self.new_like(self.tensor / value)

    def __iadd__(self, value: Tensor | float) -> None:
        self.tensor += value

    def __isub__(self, value: Tensor | float) -> None:
        self.tensor -= value

    def __imul__(self, value: Tensor | float) -> None:
        self.tensor *= value

    def __idiv__(self, value: Tensor | float) -> None:
        self.tensor /= value


class AngularField(Field):
    # NOTE: require lower bound = 0 for torch.remainder to work!
    domain: Constraint = periodic

    def __add__(self, value: Tensor | float) -> "AngularField":
        return self.new_like(torch.remainder(torch.self.tensor + value, 2 * π))

    def __sub__(self, value: Tensor | float) -> "AngularField":
        return self.new_like(torch.remainder(torch.self.tensor - value, 2 * π))

    def __iadd__(self, value: Tensor | float) -> None:
        self.tensor.add_(value).remainder_(2 * π)

    def __isub__(self, value: Tensor | float) -> None:
        self.tensor.sub_(value).remainder_(2 * π)


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
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        ...

    @abstractmethod
    def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
        ...
