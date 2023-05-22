from math import pi as π

import torch
import torch.linalg as LA

from torchlft.CPN.action import *

g = 1.0
L = 6
B = 100
N = 5


def test_actions_agree():
    x = torch.empty(B, L, L, 2 * N).normal_()
    x = x / LA.vector_norm(x, dim=-1, keepdim=True)

    z_real, z_imag = x[..., ::2], x[..., 1::2]
    z = z_real + 1j * z_imag

    A = torch.empty(B, L, L, 2).uniform_(0, 2 * π)
    λ = torch.exp(1j * A)

    a1 = action_v1(z, λ, g).real
    a2 = action_v2(z, A, g)
    a3 = action_v3(x, A, g)

    assert torch.allclose(a1, a2)
    assert torch.allclose(a2, a3)
