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

from torch import Tensor, BoolTensor, LongTensor, Size
from torch.nn import Module

from torchlft.fields import (
    Field,
    CanonicalField,
    PartitionedField,
    MaskedField,
    CompositeField,
)

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
__all__ += torchlft.fields.__all__

__all__ += [
    "Activation",
]

# probably not necessary - do i need isinstance??
@runtime_checkable
class Transform(Protocol):
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        ...

    def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
        ...


class FieldTransform(Protocol):
    def forward(self, Φ: Field) -> Field:
        ...

    def inverse(self, Ψ: Field) -> Field:
        ...


# Base density, action, hamiltonian?

# So the help isn't flooded with all available nn.Modules
# NOTE: torch master branch has __all__ attribute. When that is merged replace
# explicit list with torch.nn.modules.activations.__all__
_ACTIVATIONS = tuple(
    getattr(torch.nn, a)
    for a in [
        "ELU",
        "Hardshrink",
        "Hardsigmoid",
        "Hardtanh",
        "Hardswish",
        "LeakyReLU",
        "LogSigmoid",
        "PReLU",
        "ReLU",
        "ReLU6",
        "RReLU",
        "SELU",
        "CELU",
        "GELU",
        "Sigmoid",
        "SiLU",
        "Mish",
        "Softplus",
        "Softshrink",
        "Softsign",
        "Tanh",
        "Tanhshrink",
        "Threshold",
        "GLU",
        "Identity",
    ]
)

Activation = Union[_ACTIVATIONS]
