import pytest
import torch

from torchlft.flows.coupling import (
    CouplingLayer,
    CouplingLayerGeometryPreserving,
)
from torchlft.transforms import Translation, Rescaling, AffineTransform
import torchlft.utils

LATTICE_SHAPE = [6, 6]
MASK = torchlft.utils.make_checkerboard(LATTICE_SHAPE)


@pytest.mark.parametrize(
    "transform", [Translation(), Rescaling()]  # , AffineTransform()]
)
def test_coupling_layer(transform):

    # NOTE: should test masks of non-equal sizes
    net = lambda x: x

    layer = CouplingLayer(transform, net, MASK)

    x = torch.rand(100, *LATTICE_SHAPE)
    ldj = torch.zeros(100)

    y, ldj = layer(x, ldj)

    assert torch.allclose(y[:, ~MASK], x[:, ~MASK])

    xx, ldj = layer.inverse(y, ldj)

    assert torch.allclose(x, xx, atol=1e-5)
    assert torch.allclose(ldj, torch.zeros_like(ldj))
