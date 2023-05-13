import torch

from torchlft.typing import *
from torchlft.utils.tensor import dot, cross


def spherical_triangle_area(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    assert all([t.shape[-1] == 3 for t in (a, b, c)])

    real_part = 1 + dot(a, b) + dot(b, c) + dot(c, a)
    imag_part = dot(a, cross(b, c))

    return 2 * torch.atan2(imag_part, real_part)
