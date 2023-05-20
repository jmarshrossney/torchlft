from __future__ import annotations

from math import pi as π
from typing import TYPE_CHECKING

import torch
import torch.linalg as LA

Tensor = torch.Tensor


def dot(x: Tensor, y: Tensor) -> Tensor:
    return torch.einsum("...i,...i->...", x, y)


def outer(x: Tensor, y: Tensor) -> Tensor:
    return torch.einsum("...i,...j->...ij", x, y)


def cross(x: Tensor, y: Tensor) -> Tensor:
    return LA.cross(x, y, dim=-1)
