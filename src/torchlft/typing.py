from collections.abc import Iterable
from typing import Any, Callable, Optional, TypeAlias

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
