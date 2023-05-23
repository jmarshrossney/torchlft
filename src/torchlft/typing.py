from collections.abc import Iterable, Iterator
from typing import (
    Any,
    Callable,
    Protocol,
    Optional,
    TypeAlias,
    runtime_checkable,
    Union,
)

from torch import Tensor, BoolTensor, LongTensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

__all__ = [
    # Standard lib
    "Any",
    "Callable",
    "Iterable",
    "Iterator",
    "Optional",
    "Union",
    # PyTorch
    "Tensor",
    "BoolTensor",
    "LongTensor",
    "Optimizer",
    # Custom types
    "Scheduler",
    # Protocols
    "BaseAction",
    "Geometry",
    "Transform",
]

# Custom types

Scheduler: TypeAlias = Union[_LRScheduler, ReduceLROnPlateau]

# Protocols


class Action(Protocol):
    def compute(self, inputs: Tensor | tuple[Tensor, ...]) -> Tensor:
        ...

    def gradient(
        self, inputs: Tensor | tuple[Tensor, ...]
    ) -> Tensor | tuple[Tensor, ...]:
        ...


# NOTE: Do I need to inherit from Protocol again here?
class BaseAction(Action, Protocol):
    def sample(self, n: int) -> Tensor | tuple[Tensor, ...]:
        ...


@runtime_checkable
class Constraint(Protocol):
    def check(self, inputs: Tensor) -> bool:
        ...


class Geometry(Protocol):
    def partition(
        self, inputs: Tensor | tuple[Tensor, ...]
    ) -> tuple[Tensor, ...]:
        ...

    def restore(
        self, partitions: tuple[Tensor, ...]
    ) -> Tensor | tuple[Tensor, ...]:
        ...


class Transform(Protocol):
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        ...

    def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
        ...
