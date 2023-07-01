from typing import Any, Callable, Protocol, TypeAlias

import torch

Tensor: TypeAlias = torch.Tensor

TransformFunc: TypeAlias = Callable[
    [Tensor, Tensor, Any, ...], tuple[Tensor, Tensor]
]


class Transform(Protocol):
    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        ...

    def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
        ...
