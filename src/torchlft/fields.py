from __future__ import annotations

from abc import ABC, abstractmethod
from math import pi as π
from typing import TYPE_CHECKING

import torch

import torchlft.constraints as constraints
from torchlft.utils.lattice import assert_valid_partitioning

# from torchlft.utils.tensor import batched_mv

if TYPE_CHECKING:
    from torchlft.abc import Constraint
    from torchlft.typing import *


__all__ = [
    "CanonicalScalarField",
    "CanonicalAngularField",
    "CanonicalClassicalSpinField",
    "PartitionedScalarField",
    "PartitionedAngularField",
    "MaskedScalarField",
    "MaskedAngularField",
]

DEBUG: bool = True


class Field(ABC):
    def __init__(self, data: Tensor, *args, **metadata) -> None:

        if DEBUG:
            self.domain.check(data)

        args = [
            v.to(data.device) if isinstance(v, torch.Tensor) else v
            for v in args
        ]

        metadata |= {
            k: v.to(data.device)
            for (k, v) in metadata.items()
            if isinstance(v, torch.Tensor)
        }

        self._data = data
        self._args = args
        self._metadata = metadata

    @property
    def data(self) -> Tensor:
        return self._data

    @data.setter
    def data(self, new: Tensor) -> None:
        if DEBUG:
            self.domain.check(new)
        assert new.shape == self.data.shape
        assert new.dtype == self.data.dtype
        assert new.device == self.data.device
        self._data = new

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def device(self) -> torch.device:
        return self.data.device

    def _new_like(self, new_data: Tensor) -> Field:
        """Internal method for creating new 'similar' fields
        with different data but the same metadata.

        Internal because in a more general case checks may need to be
        run to check that the new data is compatible.
        """
        # NOTE: dict is not copied. Same data going into new
        # think about deepcopy vs shallow copy of metadata
        # Probably just shallow copy, but need to be very clear that
        # init kwargs should never be modified after instantiation
        # (which is the same as data anyway)
        return type(self)(new_data, *self._args, **self._metadata)

    def clone(self) -> Field:
        return self._new_like(self.data.clone())

    def to(self, *args, **kwargs) -> Field:
        data = self.data.to(*args, **kwargs)
        if data is self.data:
            return self
        # __init__ should move any tensors in args and metadata to correct device
        return self._new_like(data)

    @property
    @abstractmethod
    def domain(self) -> Constraint:
        ...

    @property
    @abstractmethod
    def canonical_class(self) -> type[CanonicalField]:
        ...

    @abstractmethod
    def to_canonical(self) -> CanonicalField:
        ...

    @classmethod
    @abstractmethod
    def from_canonical(cls, other: CanonicalField) -> Field:
        ...


class CanonicalField(Field):
    def __init__(self, data: Tensor, **metadata):

        shape_metadata = dict(
            batch_size=len(data),
            lattice_shape=data.shape[1 : 1 + self.lattice_dim],
            element_shape=data.shape[1 + self.lattice_dim :],
        )
        if any(
            [
                metadata.get(key) not in (None, shape_metadata[key])
                for key in ("batch_size", "lattice_shape", "element_shape")
            ]
        ):
            raise ValueError

        metadata |= shape_metadata

        super().__init__(data, **metadata)

    @property
    def canonical_class(self):
        return type(self)

    def to_canonical(self):
        return self

    @classmethod
    def from_canonical(cls, other: CanonicalField) -> CanonicalField:
        assert (
            type(other) is cls
        ), f"Type mismatch: expected {cls} but got {type(other)}"
        return other

    @property
    def batch_size(self) -> int:
        return self.metadata["batch_size"]

    @property
    @abstractmethod
    def lattice_shape(self) -> torch.Size:
        self.metadata["lattice_shape"]

    @property
    @abstractmethod
    def element_shape(self) -> torch.Size:
        self.metadata["element_shape"]


class PartitionedField(Field):
    def __init__(self, data: Tensor, lattice_shape: torch.Size, **metadata):
        super().__init__(data, lattice_shape, **metadata)

    @property
    def lattice_shape(self) -> torch.Size:
        return self._args[0]

    @property
    def batch_size(self) -> int:
        return self.data.shape[1]

    @classmethod
    def from_canonical(cls, other: CanonicalField) -> Field:
        masks = cls.get_masks(other.lattice_shape, other.device)

        assert_valid_partitioning(masks)

        partitions = [other.data[:, mask] for mask in masks]

        return cls(
            torch.stack(partitions),
            lattice_shape=other.lattice_shape,
        )

    def to_canonical(self) -> CanonicalField:
        data_canonical = torch.empty(
            (self.batch_size, *self.lattice_shape),
            device=self.data.device,
            dtype=self.data.dtype,
        )

        masks = self.get_masks(self.lattice_shape, self.device)
        assert_valid_partitioning(masks)

        for partition, mask in zip(self.data, masks):
            data_canonical[:, mask] = partition

        return self.canonical_class(data_canonical, **self.metadata)

    @staticmethod
    @abstractmethod
    def get_masks(
        lattice_shape: torch.Size, device: torch.device
    ) -> Iterable[BoolTensor]:
        ...


class MaskedField(Field):
    # TODO: something with domain so that it knows to expect NaN

    def __init__(self, data: Tensor, lattice_shape: torch.Size, **metadata):
        super().__init__(data, lattice_shape, **metadata)

    @property
    def lattice_shape(self) -> torch.Size:
        return self._args[0]

    @Field.data.setter
    def data(self, new: Tensor):
        assert torch.equal(new_data.isnan(), self.data.isnan())
        Field.data.fset(new)

    @classmethod
    def from_canonical(cls, other: CanonicalField) -> Field:
        masks = self.get_masks(other.lattice_shape, other.device)
        assert_valid_partitioning(masks)

        # Use NaN so that usual operations (+-*/ etc) still work
        partitions = [
            other.data.masked_fill(mask, float("nan")) for mask in masks
        ]

        return cls(
            torch.stack(partitions),
            lattice_shape=other.lattice_shape,
        )

    def to_canonical(self) -> CanonicalField:
        return self.canonical_class(
            sum(self.data.nan_to_num()), **self.metadata
        )

    @staticmethod
    @abstractmethod
    def get_masks(
        lattice_shape: torch.Size, device: torch.device
    ) -> Iterable[BoolTensor]:
        ...


class CompositeField:
    def __init__(self, *fields: Field, **metadata):
        assert len(set([field.device for field in fields])) == 1

        self._fields = fields
        self._metadata = metadata

    def __iter__(self):
        return iter(self._fields)

    @property
    def data(self) -> tuple[Tensor, ...]:
        return (field.data for field in self)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def device(self) -> torch.device:
        return self.data[0].device

    def clone(self) -> CompositeField:
        cloned_fields = [field.clone for field in self]
        return type(self)(*cloned_fields, **self._metadata)

    def to(self, *args, **kwargs) -> CompositeField:
        new_fields = [field.to(*args, **kwargs) for field in self]
        if all([new is current for (new, current) in zip(new_fields, self)]):
            return self
        return type(self)(*new_fields, **self._metadata)

    @property
    @abstractmethod
    def canonical_class(self) -> type[CanonicalField]:
        ...

    @abstractmethod
    def to_canonical(self) -> CanonicalField:
        ...

    @classmethod
    @abstractmethod
    def from_canonical(cls, other: CanonicalField) -> Field:
        ...


Field.register(CompositeField)


class _ScalarFieldMixin:
    def __pos__(self) -> Field:
        return self

    def __neg__(self) -> Field:
        return self._new_like(-self.data)

    def __add__(self, value: Tensor | float) -> Field:
        return self._new_like(self.data + value)

    def __sub__(self, value: Tensor | float) -> Field:
        return self._new_like(self.data - value)

    def __mul__(self, value: Tensor | float) -> Field:
        return self._new_like(self.data * value)

    def __div__(self, value: Tensor | float) -> Field:
        return self._new_like(self.data / value)

    def __iadd__(self, value: Tensor | float) -> None:
        self.data += value

    def __isub__(self, value: Tensor | float) -> None:
        self.data -= value

    def __imul__(self, value: Tensor | float) -> None:
        self.data *= value

    def __idiv__(self, value: Tensor | float) -> None:
        self.data /= value


class _AngularFieldMixin:
    # NOTE: require lower bound = 0 for torch.remainder to work!
    domain: Constraint = constraints.periodic

    def __add__(self, value: Tensor | float) -> Field:
        return self._new_like(torch.remainder(torch.self.data + value, 2 * π))

    def __sub__(self, value: Tensor | float) -> Field:
        return self._new_like(torch.remainder(torch.self.data - value, 2 * π))

    def __iadd__(self, value: Tensor | float) -> None:
        self.data.add_(value).remainder_(2 * π)

    def __isub__(self, value: Tensor | float) -> None:
        self.data.sub_(value).remainder_(2 * π)


class CanonicalScalarField(CanonicalField, _ScalarFieldMixin):
    domain = constraints.real

    @property
    def lattice_shape(self) -> torch.Size:
        return self.data.shape[1:]

    @property
    def element_shape(self) -> torch.Size:
        return torch.Size([])


class CanonicalAngularField(CanonicalField, _AngularFieldMixin):
    @property
    def lattice_shape(self) -> torch.Size:
        return self.data.shape[1:]

    @property
    def element_shape(self) -> torch.Size:
        return torch.Size([])


class CanonicalClassicalSpinField(CanonicalField):
    @property
    def lattice_shape(self) -> torch.Size:
        return self.data.shape[1:-1]

    @property
    def element_shape(self) -> torch.Size:
        return self.data.shape[-1:]

    def __rmatmul__(self, other: Tensor) -> Tensor:
        raise NotImplementedError
        # NOTE: probably useless
        # rotated = batched_mv(other, self.data)
        # return self.new_like(rotated)


# Have to do this after defining the canonical classes
_ScalarFieldMixin.canonical_class = CanonicalScalarField
_AngularFieldMixin.canonical_class = CanonicalAngularField

# Not working
class ScalarField(Field):
    @classmethod
    def __subclasscheck__(cls, C):
        print("hi")
        return (
            issubclass(C, Field) and C.canonical_class is CanonicalScalarField
        )
