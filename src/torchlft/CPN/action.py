from __future__ import annotations

import torch

from torchlft.utils.linalg import dot, mv
from torchlft.utils.tensor import sum_except_batch


def action_v1(z: Tensor, λ: Tensor, g: float) -> Tensor:
    assert z.dim() == 4
    assert λ.dim() == 4
    assert λ.shape[:-1] == z.shape[:-1]
    assert λ.shape[-1] == 2

    # Dμ_z = λ.unsqueeze(-1).conj() * torch.stack([z.roll(-1, 1), z.roll(-1, 2), dim=-2]) - z.unsqueeze(-2)

    s = torch.zeros_like(λ).sum(dim=-1)

    for μ, λ_μ in zip((1, 2), λ.split(1, -1)):
        Dμ_z = λ_μ.conj() * z.roll(-1, μ) - z

        s += dot(Dμ_z, Dμ_z.conj())

    return (1 / g) * sum_except_batch(s)


def action_v2(z: Tensor, A: Tensor, g: float) -> Tensor:
    assert z.dim() == 4
    assert A.dim() == 4
    assert A.shape[:-1] == z.shape[:-1]
    assert A.shape[-1] == 2

    s = torch.zeros_like(A).sum(dim=-1)

    for μ, A_μ in zip((1, 2), A.split(1, -1)):
        A_μ.squeeze_(-1)
        zz_μ = dot(z.conj(), z.roll(-1, μ))
        r_μ, θ_μ = zz_μ.abs(), zz_μ.angle()
        s += 1 - r_μ * torch.cos(A_μ - θ_μ)

    return (2 / g) * sum_except_batch(s)


def action_v3(x: Tensor, A: Tensor, g: float) -> Tensor:
    assert x.dim() == 4
    assert A.dim() == 4
    assert A.shape[:-1] == x.shape[:-1]
    assert A.shape[-1] == 2

    assert x.shape[-1] % 2 == 0
    N = x.shape[-1] // 2

    s = torch.zeros_like(A).sum(dim=-1)

    for μ, A_μ in zip((1, 2), A.split(1, -1)):
        A_μ.unsqueeze_(-1)
        J_μ = torch.kron(
            torch.eye(N),
            A_μ.cos() * torch.eye(2)
            + A_μ.sin() * torch.tensor([[0, -1], [1, 0]]),
        )
        assert J_μ.shape == (*x.shape, 2 * N)

        x_μ = x.roll(-1, μ)
        first = dot(x, mv(J_μ.transpose(-2, -1), x_μ))
        second = dot(x_μ, mv(J_μ, x))
        s += 2 - first - second
        """
        # This is wrong. I also don't see the point right now
        Jx = mv(J_μ.transpose(-2, -1), x.roll(-1, μ)) + mv(
            J_μ.roll(+1, μ), x.roll(+1, μ)
        )
        xJx = dot(x, Jx)
        s += 4 - xJx
        """

    return (1 / g) * sum_except_batch(s)


def action_v4(z: Tensor, g: float) -> Tensor:
    s = torch.zeros_like(z).sum(dim=-1)

    for μ in (1, 2):
        s += 1 - dot(z.roll(-1, μ).conj(), z).abs().pow(2)

    return (1 / g) * sum_except_batch(s)
