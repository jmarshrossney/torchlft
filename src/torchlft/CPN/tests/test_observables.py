import torch
import torch.linalg as LA

from torchlft.CPN.observables import *
from torchlft.utils.linalg import dot, outer, mm, tr


L = 500
B = 10
N = 4


def _test_linalg():
    x = torch.empty(2 * N).normal_()
    x = x / LA.vector_norm(x, dim=-1, keepdim=True)

    z_real, z_imag = x[::2], x[1::2]
    z = z_real + 1j * z_imag

    P = outer(z, z.conj())

    z_dagger_z = dot(z.conj(), z)

    assert torch.allclose(
        z_dagger_z,
        torch.einsum("ii", P),  # tr(P)
    )

    trPP = torch.einsum("ij,ji", P, P)
    print(trPP)

    assert torch.allclose(
        z_dagger_z.real**2 + z_dagger_z.imag**2,
        torch.einsum("ij,ji", P, P).real,  # tr(PP)
    )

    A, B = torch.rand(10, 10), torch.rand(10, 10)
    assert torch.allclose(
        tr(torch.mm(A, B)),
        torch.einsum("ij,ji", A, B),
    )


def test_trace_three_matrices():
    A, B, C = [torch.rand(10, 10) for _ in range(3)]

    long_way = tr(mm(A, mm(B, C)))
    short_way = torch.einsum("ij,jk,ki", A, B, C)
    assert torch.allclose(long_way, short_way)


def test_correlators_agree():
    x = torch.empty(B, L, L, 2 * N).normal_()
    x = x / LA.vector_norm(x, dim=-1, keepdim=True)

    z_real, z_imag = x[..., ::2], x[..., 1::2]
    z = z_real + 1j * z_imag

    c1 = two_point_correlator(z)
    c2 = two_point_correlator_v2(z)

    c2_real, c2_imag = c2.real, c2.imag

    assert torch.allclose(c2_imag, torch.zeros(1))
    assert torch.allclose(c1, c2_real)


def test_topological_charge_geometric():
    x = torch.empty(B, L, L, 2 * N).normal_()
    x = x / LA.vector_norm(x, dim=-1, keepdim=True)

    z_real, z_imag = x[..., ::2], x[..., 1::2]
    z = z_real + 1j * z_imag

    Q = topological_charge_geometric(z)

    print(charge)
    assert torch.allclose(Q, Q.round())

def test_topological_charge_v2():

    x = torch.empty(B, L, L, 2 * N).normal_()
    x = x / LA.vector_norm(x, dim=-1, keepdim=True)

    z_real, z_imag = x[..., ::2], x[..., 1::2]
    z = z_real + 1j * z_imag

    Q = topological_charge_v2(z)

    print(charge)
    assert torch.allclose(Q, Q.round())


def test_topological_charge_v3():

    A = torch.empty(B, L, L, 2).uniform_(0, 2 * π)

    Q = topological_charge_v3(A)

    print(charge)
    assert torch.allclose(Q, Q.round())


def test_topological_charge_similar():
    x = torch.empty(B, L, L, 2 * N).normal_()
    x = x / LA.vector_norm(x, dim=-1, keepdim=True)

    z_real, z_imag = x[..., ::2], x[..., 1::2]
    z = z_real + 1j * z_imag

    Q1 = topological_charge_geometric(z)
    Q2 = topological_charge_v2(z)

    print("")
    print(Q1)
    print(Q2)
