from collections.abc import Iterable
from typing import (
    Any,
    Callable,
    Optional,
    Protocol,
    runtime_checkable,
    TypeAlias,
    TYPE_CHECKING,
    Union,
)

import torch
from torch import Tensor, BoolTensor, LongTensor, Size
from torch.nn import Module

__all__ = [
    "Any",
    "Callable",
    "Iterable",
    "Optional",
    "TypeAlias",
]

__all__ += [
    "Tensor",
    "BoolTensor",
    "LongTensor",
    "Size",
    "Module",
]

# Custom types


# probably not necessary - do i need isinstance??
@runtime_checkable
class Transform(Protocol):
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        ...

    def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
        ...


# Base density, action, hamiltonian?
