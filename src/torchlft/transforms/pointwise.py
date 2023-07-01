from typing import ClassVar, Callable, TypeAlias

import torch

Tensor: TypeAlias = torch.Tensor

def pointwise_transform_(
        forward_fn: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
        inverse_fn: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
        handle_params_fn: Callable[Tensor | tuple[Tensor, ...], Tensor | tuple[Tensor, ...]],
        n_params: int,
):
    _n_params = n_params

    class PointwiseTransform:
        n_params: ClassVar[int] = _n_params

        def __init__(self, params: Tensor):
            self.params = handle_params_fn(params)

        def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
            return forward_fn(x, self.params)

        def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
            return inverse_fn(y, self.params)

    return PointwiseTransform
