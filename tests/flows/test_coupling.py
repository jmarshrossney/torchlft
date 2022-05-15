import pytest
import torch

import torchlft.flows.coupling
from torchlft.transforms import Translation, Rescaling, AffineTransform
import torchlft.utils

LATTICE_SHAPE = [6, 6]
MASK = torchlft.utils.make_checkerboard(LATTICE_SHAPE)


class CouplingLayer(torchlft.flows.coupling.CouplingLayer):
    def net_forward(self, x_masked):
        return torch.stack(
            [
                torch.tanh(x_masked)  # some deterministic transform
                for _ in range(self._transform.n_params)
            ],
            dim=1,
        )


@pytest.mark.parametrize(
    "transform", [Translation(), Rescaling(), AffineTransform()]
)
def test_coupling_layer(transform):

    layer = CouplingLayer(transform, MASK)

    x = torch.rand(100, *LATTICE_SHAPE)
    ldj = torch.zeros(100)

    y, ldj = layer(x, ldj)

    assert torch.allclose(y[:, ~MASK], x[:, ~MASK])

    xx, ldj = layer.inverse(y, ldj)

    assert torch.allclose(x, xx, atol=1e-5)
    assert torch.allclose(ldj, torch.zeros_like(ldj))
