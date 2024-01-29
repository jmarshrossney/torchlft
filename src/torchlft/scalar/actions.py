from typing import TypeAlias

import torch

Tensor: TypeAlias = torch.Tensor


class FreeScalarAction:
    def __init__(self, m_sq: float):
        self.m_sq = m_sq

    def __call__(self, φ: Tensor) -> Tensor:
        s = torch.zeros_like(φ)

        for μ in (1, 2):  # TODO: accept different dims?
            s -= φ * φ.roll(-1, μ)

        s += 0.5 * (4 + self.m_sq) * φ**2

        return s.sum(dim=(1, 2))

    def grad(self, φ: Tensor) -> Tensor:
        dsdφ = torch.zeros_like(φ)

        for μ in (1, 2):
            dsdφ -= φ.roll(-1, μ) + φ.roll(+1, μ)

        dsdφ += (4 + self.m_sq) * φ

        return dsdφ
