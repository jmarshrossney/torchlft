from __future__ import annotations

from math import pi as π
from collections.abc import Iterable

import torch

Tensor = torch.Tensor


def mod_2pi(x: Tensor) -> Tensor:
    return torch.remainder(x, 2 * π)


def expand_like_stack(x: Tensor, n: int) -> Tensor:
    expanded_shape = [n] + list(x.shape)
    return x.expand(expanded_shape)


def expand_elements(
    x: Tensor, target_shape: torch.Size, stack_dim: int = 0
) -> torch.Tensor:
    """
    Expands and stacks each element of a one-dimensional tensor.

    The input tensor is split into chunks of size 1 along the zeroth
    dimension, each element expanded, and then the result stacked.

    Args:
        x
            One-dimensional input tensor
        target_shape
            Shape that each element will be expanded to
        stack_dim
            Dimension that the resulting expanded elements will be
            stacked on

    Effectively this does the following:

    .. code-block:: python

        elements = x.split(1)
        elements = [el.expand(target_shape) for el in elements]
        out = torch.stack(elements, dim=stack_dim)
        return out
    """
    return torch.stack(
        [el.expand(target_shape) for el in x.split(1)],
        dim=stack_dim,
    )


def stacked_nan_to_num(
    x: torch.Tensor, y: torch.Tensor, dim: int
) -> torch.Tensor:
    """
    Vector-wise replacement of NaNs.

    Args:
        x:
            The tensor with NaN values
        y:
            One-dimensional tensor to replace the NaNs with
        dim:
            Dimension of ``x`` to fill with ``y``

    Example:

        >>> x = torch.rand(10, 2, 2)
        >>> x[x < 0.5] = float('nan')
        >>> y = torch.tensor([10, 20])
        >>> out = stacked_nan_to_num(x, y, dim=1)
        >>> x[0], out[0]
        (tensor([[nan, nan],
                 [nan, nan]]),
         tensor([[10., 10.],
                 [20., 20.]]))
        >>> x[1], out[1]
        (tensor([[nan, 0.9123],
                 [0.5999, nan]]),
         tensor([[10.0000, 0.9213.],
                 [0.5999., 20.0000]]))
        >>> out = stacked_nan_to_num(x, y, dim=2)
        >>> out[0]
        tensor([[10., 20.],
                [10., 20.]])
    """
    assert y.dim() == 1, "Only one-dimensional replacements supported"
    # Expand y to make x & y broadcastable
    if -1 < dim < x.dim() - 1:  # anything but last dim
        y = expand_elements(y, x.shape[dim + 1 :], stack_dim=0)  # noqa: E203
    return x.nan_to_num(0).add(y.mul(x.isnan()))


def sum_except_batch(x: Tensor) -> Tensor:
    """
    Sum over all but the first dimension of the input tensor.
    """
    return x.flatten(start_dim=1).sum(dim=1)


def tuple_concat(tuples: Iterable[tuple[Tensor, ...]]) -> tuple[Tensor, ...]:
    """
    Dim 0 concatenation of tuples of tensors.

    Example:

        >>> a, b, c = (
                (torch.rand(1), torch.rand(10), torch.rand(1, 10))
                for _ in range(3)
        )
        >>> x, y, z = tuple_concat([a, b, c])
        >>> x.shape, y.shape, z.shape
        (torch.Size([3]), torch.Size([30]), torch.Size([3, 10]))

    """
    return (torch.cat(tensors) for tensors in map(list, zip(*tuples)))


def dict_concat(
    dicts: Iterable[dict[str, Tensor]], strict: bool = True
) -> dict[str, Tensor]:
    """
    Dim-zero concatenation of dicts of torch.Tensors.

    Examples:

        Dicts with matching keys:

        >>> a, b, c = (
                {"x": torch.rand(1), "y": torch.rand(10), z: torch.rand(1, 10)}
                for _ in range(3)
        )
        >>> out = dict_concat(a, b, c)
        >>> out["x"].shape, out["y"].shape, out["z"].shape
        (torch.Size([3]), torch.Size([30]), torch.Size([3, 10]))

        Dicts without matching keys:

        >>> d = {"x": torch.rand(1), "w": torch.rand(100)}
        >>> dict_concat(a, b, c, d)
        KeyError: 'y'
        >>> out = dict_concat(a, b, c, d, strict=False)
        >>> out["x"].shape, out["y"].shape, out["z"].shape, out["w"].shape
        (torch.Size([4]), torch.Size([30]), torch.Size([3, 10]), torch.Size([100]))  # noqa: E501
    """
    keys = set([k for d in dicts for k in d.keys()])
    if strict:
        return {k: torch.cat([d[k] for d in dicts]) for k in keys}
    else:
        return {
            k: torch.cat([d.get(k, torch.tensor([])) for d in dicts])
            for k in keys
        }
