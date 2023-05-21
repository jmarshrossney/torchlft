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


def mv(M: Tensor, v: Tensor) -> Tensor:
    return torch.einsum("...ij,...j->...i", M, v)


def vm(v: Tensor, M: Tensor) -> Tensor:
    return torch.einsum("...i,...ij->...j", v, M)


def projector(x: Tensor) -> Tensor:
    return torch.eye(x.shape[-1], dtype=x.dtype, device=x.device) - outer(
        x, x
    ) / dot(x, x).unflatten(-1, (-1, 1, 1))


def orthogonal_projection(v: Tensor, x: Tensor) -> Tensor:
    return mv(projector(x), v)
