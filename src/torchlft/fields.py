from abc import ABC, abstractmethod
from math import pi as π

import torch

from torchlft.constraints import (
    _Constraint,
    real,
    periodic,
)
from torchlft.typing import *

__all__ = [
    "CanonicalScalarField",
    "CanonicalPeriodicScalarField",
]

DEBUG: bool = True


class _Field(ABC):
    def __init__(
        self,
        data,
        *,
        lattice_shape: torch.Size,
        element_shape: torch.Size,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
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

    def clone(self) -> "_Field":
        return self.new_like(self.tensor.clone())

    @property
    @abstractmethod
    def domain(self) -> _Constraint:
        ...

    @abstractmethod
    def to_canonical(self) -> "_Field":
        ...

    @classmethod
    @abstractmethod
    def from_canonical(cls, other: "_Field") -> "_Field":
        ...

    def new_like(self, data: Tensor) -> "_Field":
        ...


class _ScalarField(_Field):
    domain: _Constraint = real

    def __pos__(self) -> "_ScalarField":
        return self

    def __neg__(self) -> "_ScalarFielf":
        return self.new_like(-self.tensor)

    def __add__(self, value: Tensor | float) -> "_ScalarField":
        return self.new_like(self.tensor + value)

    def __sub__(self, value: Tensor | float) -> "_ScalarField":
        return self.new_like(self.tensor - value)

    def __mul__(self, value: Tensor | float) -> "_ScalarField":
        return self.new_like(self.tensor * value)

    def __div__(self, value: Tensor | float) -> "_ScalarField":
        return self.new_like(self.tensor / value)

    def __iadd__(self, value: Tensor | float) -> None:
        self.tensor += value

    def __isub__(self, value: Tensor | float) -> None:
        self.tensor -= value

    def __imul__(self, value: Tensor | float) -> None:
        self.tensor *= value

    def __idiv__(self, value: Tensor | float) -> None:
        self.tensor /= value


class _PeriodicScalarField(_Field):
    # NOTE: require lower bound = 0 for torch.remainder to work!
    domain: _Constraint = periodic

    def __add__(self, value: Tensor | float) -> "_ScalarField":
        return self.new_like(torch.remainder(torch.self.tensor + value, 2 * π))

    def __sub__(self, value: Tensor | float) -> "_ScalarField":
        return self.new_like(torch.remainder(torch.self.tensor - value, 2 * π))

    def __iadd__(self, value: Tensor | float) -> None:
        self.tensor.add_(value).remainder_(2 * π)

    def __isub__(self, value: Tensor | float) -> None:
        self.tensor.sub_(value).remainder_(2 * π)


class CanonicalScalarField(_ScalarField):
    def __init__(
        self,
        data,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:

        tensor = torch.as_tensor(data, dtype=dtype, device=device)
        _, *lattice_shape = tensor.shape

        super().__init__(
            data,
            lattice_shape=lattice_shape,
            element_shape=(1,),
            dtype=dtype,
            device=device,
        )

    def to_canonical(self) -> "CanonicalScalarField":
        return self

    @classmethod
    def from_canonical(
        cls, other: "CanonicalScalarField"
    ) -> "CanonicalScalarField":
        assert (
            type(other) is cls
        ), f"Type mismatch: expected {cls} but got {type(other)}"
        return other

    def new_like(self, data: Tensor) -> "CanonicalScalarField":
        ...


class CanonicalPeriodicScalarField(_PeriodicScalarField):
    def __init__(
        self,
        data,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:

        tensor = torch.as_tensor(data, dtype=dtype, device=device)
        _, *lattice_shape = tensor.shape

        super().__init__(
            data,
            lattice_shape=lattice_shape,
            element_shape=(1,),
            dtype=dtype,
            device=device,
        )

    def to_canonical(self) -> "CanonicalPeriodicScalarField":
        return self

    @classmethod
    def from_canonical(
        cls, other: "CanonicalPeriodicScalarField"
    ) -> "CanonicalPeriodicScalarField":
        assert (
            type(other) is cls
        ), f"Type mismatch: expected {cls} but got {type(other)}"
        return other


class CanonicalClassicalSpinField(_Field):
    def __init__(
        self,
        data,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:

        tensor = torch.as_tensor(data, dtype=dtype, device=device)
        _, *lattice_shape, vector_dim = tensor.shape

        super().__init__(
            data,
            lattice_shape=lattice_shape,
            element_shape=(vector_dim,),
            dtype=dtype,
            device=device,
        )

    def __rmatmul__(self, other: Tensor) -> Tensor:
        rotated = batched_mv(other, self.tensor)
        return self.new_like(rotated)

    def to_canonical(self) -> "CanonicalPeriodicScalarField":
        return self

    @classmethod
    def from_canonical(
        cls, other: "CanonicalClassicalSpinField"
    ) -> "CanonicalClassicalSpinField":
        assert (
            type(other) is cls
        ), f"Type mismatch: expected {cls} but got {type(other)}"
        return other


class ComplexScalarField:
    ...


class U1Field:
    ...


class U1FieldComplexRep:
    ...
