import torch

from torchlft.phi_four.actions import (
    phi_four_action,
    PhiFourAction,
)


def test_all_equal_no_quart():
    beta = 1
    m_sq = -2
    lamda = 0

    x = torch.rand(100, 10, 10)

    s_func = phi_four_action(x, m_sq=m_sq, lamda=lamda)
    i_func = phi_four_action(x, beta=beta, lamda=lamda)
    s_cls = PhiFourAction(m_sq=m_sq, lamda=lamda)(x)
    i_cls = PhiFourAction(beta=beta, lamda=lamda)(x)

    assert torch.allclose(s_func, s_cls)
    assert torch.allclose(i_func, i_cls)
    assert torch.allclose(s_func, i_func)
    assert torch.allclose(s_cls, i_cls)
