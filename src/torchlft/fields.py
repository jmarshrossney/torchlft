from math import pi as π

import torch

from torchlft.abc import Field, ScalarField, AngularField
from torchlft.constraints import (
    _Constraint,
    real,
    periodic,
)
from torchlft.typing import *
from torchlft.utils.tensor import batched_mv

__all__ = [
    "CanonicalScalarField",
    "CanonicalAngularField",
    "CanonicalClassicalSpinField",
]

DEBUG: bool = True


class _CanonicalFieldMixin:
    def to_canonical(self):
        return self

    @classmethod
    def from_canonical(
        cls, other: "CanonicalScalarField"
    ) -> "CanonicalScalarField":
        assert (
            type(other) is cls
        ), f"Type mismatch: expected {cls} but got {type(other)}"
        return other


class CanonicalScalarField(ScalarField, _CanonicalFieldMixin):
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

    def new_like(self, data: Tensor) -> "CanonicalScalarField":
        # TODO: check lattice and elements match?
        # Does "new_like" imply same lattice?
        return type(self)(data, dtype=self.dtype, device=self.device)


class CanonicalAngularField(AngularField, _CanonicalFieldMixin):
    def __init__(
        self,
        data,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
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

    def new_like(self, data: Tensor) -> "CanonicalAngularField":
        return type(self)(data, dtype=self.dtype, device=self.device)


class CanonicalClassicalSpinField(Field, _CanonicalFieldMixin):
    def __init__(
        self,
        data,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
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

    def new_like(self, data: Tensor) -> "CanonicalClassicalSpinField":
        return type(self)(data, dtype=self.dtype, device=self.device)


class ComplexScalarField:
    ...


class U1Field:
    ...


class U1FieldComplexRep:
    ...
