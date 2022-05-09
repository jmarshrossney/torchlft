from __future__ import annotations

import torch

import torchlft.utils

def test_nn_kernel():
    expected = torch.Tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    got = torchlft.utils.nearest_neighbour_kernel(lattice_dim=2)
    assert torch.all(torch.eq(expected, got))
