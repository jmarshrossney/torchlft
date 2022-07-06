import torch

from torchlft.phi_four.actions import (
    phi_four_action_standard,
    phi_four_action_ising,
    PhiFourActionStandard,
    PhiFourActionIsing,
)


def test_all_equal_no_quart():
    beta = 1
    m_sq = -2
    lam = 0

    x = torch.rand(100, 10, 10)

    s_func = phi_four_action_standard(x, m_sq, lam)
    i_func = phi_four_action_ising(x, beta, lam)
    s_cls = PhiFourActionStandard(m_sq, lam)(x)
    i_cls = PhiFourActionIsing(beta, lam)(x)

    assert torch.allclose(s_func, s_cls)
    assert torch.allclose(i_func, i_cls)
    assert torch.allclose(s_func, i_func)
    assert torch.allclose(s_cls, i_cls)
