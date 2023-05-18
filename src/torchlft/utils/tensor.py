from __future__ import annotations

from math import pi as π
from typing import TYPE_CHECKING

import torch
import torch.linalg as LA

if TYPE_CHECKING:
    from torchlft.typing import *


def dot(x: Tensor, y: Tensor) -> Tensor:
    return torch.einsum("...i,...i->...", x, y)


def outer(x: Tensor, y: Tensor) -> Tensor:
    return torch.einsum("...i,...j->...ij", x, y)


# NOTE: possibly belongs in geometry
def cross(x: Tensor, y: Tensor) -> Tensor:
    return LA.cross(x, y, dim=-1)


def mod_2pi(x: Tensor) -> Tensor:
    return torch.remainder(x, 2 * π)


def sum_except_batch(x: Tensor) -> Tensor:
    return x.flatten(start_dim=1).sum(dim=1)


def expand_like_stack(x: Tensor, n: int) -> Tensor:
    expanded_shape = [n] + list(x.shape)
    return x.expand(expanded_shape)
