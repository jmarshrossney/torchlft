import torch

from torchlft.utils.linalg import dot, cross

from torchlft.typing import Tensor


def spherical_triangle_area(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    assert all([t.shape[-1] == 3 for t in (a, b, c)])

    real_part = 1 + dot(a, b) + dot(b, c) + dot(c, a)
    imag_part = dot(a, cross(b, c))

    return 2 * torch.atan2(imag_part, real_part)
