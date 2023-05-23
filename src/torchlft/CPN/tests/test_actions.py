from math import pi as π

import torch
import torch.linalg as LA

from torchlft.CPN.action import *
from torchlft.utils.linalg import orthogonal_projection

g = 1.0
L = 6
B = 100
N = 4


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


def test_action_gradient():
    x = torch.empty(B, L, L, 2 * N).normal_()
    x = x / LA.vector_norm(x, dim=-1, keepdim=True)

    A = torch.empty(B, L, L, 2).uniform_(0, 2 * π)

    # Use double rather than raising allclose tol?
    x = x.double()
    A = A.double()

    with torch.enable_grad():
        x.requires_grad_(True)
        A.requires_grad_(True)
        a3 = action_v3(x, A, g)

    g3 = grad_action_v3(x, A, g)

    a, b = g3

    g3_autograd = torch.autograd.grad(
        inputs=(x, A), outputs=a3, grad_outputs=torch.ones_like(a3)
    )

    c, d = g3_autograd

    c = orthogonal_projection(c, x)

    print("")
    print(a)
    print("")
    print(c)
    print("")
    print(b)
    print("")
    print(d)

    assert torch.allclose(a, c)
    assert torch.allclose(b, d)
