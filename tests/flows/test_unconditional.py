import torch

from torchlft.flows.unconditional import (
    UnconditionalLayer,
    GlobalRescalingLayer,
)
from torchlft.transforms import AffineTransform


def test_unconditional():
    layer = UnconditionalLayer(AffineTransform())
    x = torch.empty([4, 6, 2]).normal_()
    ldj = torch.zeros(4)

    y, ldj_fwd = layer.forward(x, ldj)

    assert torch.allclose(x, y)
    assert torch.allclose(ldj_fwd, torch.zeros_like(ldj_fwd))

    xx, ldj_inv = layer.inverse(y, ldj_fwd)
