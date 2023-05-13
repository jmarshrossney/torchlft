from math import pi as π

import torch
import torch.linalg as LA

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
