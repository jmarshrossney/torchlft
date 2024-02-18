import torch

from torchlft.scalar.action import ActionV1, ActionV2


def test_actions_agree():
    d = 2
    L = 6
    m_sq = 1.5

    φ = torch.empty(1000, L, L, 1).normal_()

    S1 = ActionV1(L, m_sq, d)(φ.flatten(1))
    S2 = ActionV2(m_sq, d)(φ)

    assert S1.shape == S2.shape
    assert torch.allclose(S1, S2, atol=1e-6)
