import math
from hypothesis import assume, given, settings, strategies
import torch

from torchlft.phi_four.actions import (
    phi_four_action,
    phi_four_action_local,
    PhiFourAction,
)
from torchlft.sample.utils import build_neighbour_list


def _test_all_equal_no_quart():
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


@given(
    lattice_shape=strategies.lists(
        strategies.integers(2, 4), min_size=1, max_size=4
    )
)
@settings(max_examples=10)
def test_local_global_actions(lattice_shape):
    phi = torch.randn([1, *lattice_shape])
    phi_flat = phi.view(-1)

    neighbour_list = build_neighbour_list(lattice_shape)

    action_glob = phi_four_action(phi, m_sq=1, lamda=1)
    action_loc = sum(
        [
            phi_four_action_local(
                phi_flat[idx], phi_flat[neighbour_list[idx]], m_sq=1, lamda=1
            )
            for idx in range(math.prod(lattice_shape))
        ]
    )

    assert torch.allclose(action_glob, action_loc)
