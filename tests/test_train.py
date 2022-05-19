import torch
import math
from pprint import pprint

from torchlft.distributions import Prior
from torchlft.flows.base import Flow
from torchlft.flows.coupling import CouplingLayer
from torchlft.flows.unconditional import UnconditionalLayer as ULayer, GlobalRescalingLayer
from torchlft.transforms import AffineTransform, Translation
import torchlft.utils

LATTICE_SHAPE = [4, 4]


def _train_loop(prior, target, flow, n_steps=300):
    optimizer = torch.optim.Adam(flow.parameters(), lr=0.05)
    for _ in range(n_steps):
        z, log_prob_z = next(prior)
        phi, log_prob_phi = flow(z, log_prob_z)
        log_weights = target.log_prob(phi).flatten(
            start_dim=1
        ).sum(dim=1) - log_prob_phi
        loss = log_weights.mean().neg()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return float(loss)

def test_shift():
    prior = Prior(
        distribution=torch.distributions.Normal(
            loc=torch.zeros(LATTICE_SHAPE).unsqueeze(0),
            scale=torch.ones(LATTICE_SHAPE).unsqueeze(0),
        ),
        batch_size=100,
    )
    target = torch.distributions.Normal(
        loc=torch.ones(LATTICE_SHAPE).unsqueeze(0),
        scale=torch.ones(LATTICE_SHAPE).unsqueeze(0),
    )

    flow = ULayer(Translation())

    loss = _train_loop(prior, target, flow)

    assert math.isclose(loss, 0, abs_tol=0.1)

def test_rescale():
    prior = Prior(
        distribution=torch.distributions.Normal(
            loc=torch.zeros(LATTICE_SHAPE).unsqueeze(0),
            scale=torch.ones(LATTICE_SHAPE).unsqueeze(0),
        ),
        batch_size=100,
    )
    target = torch.distributions.Normal(
        loc=torch.zeros(LATTICE_SHAPE).unsqueeze(0),
        scale=torch.full(LATTICE_SHAPE, fill_value=0.5).unsqueeze(0),
    )

    flow = GlobalRescalingLayer()

    loss = _train_loop(prior, target, flow)

    assert math.isclose(loss, 0, abs_tol=0.1)

def test_shift_and_rescale():
    prior = Prior(
        distribution=torch.distributions.Normal(
            loc=torch.zeros(LATTICE_SHAPE).unsqueeze(0),
            scale=torch.ones(LATTICE_SHAPE).unsqueeze(0),
        ),
        batch_size=100,
    )
    target = torch.distributions.Normal(
        loc=torch.full(LATTICE_SHAPE, fill_value=1.0).unsqueeze(0),
        scale=torch.full(LATTICE_SHAPE, fill_value=0.5).unsqueeze(0),
    )

    flow = ULayer(AffineTransform())

    loss = _train_loop(prior, target, flow)

    assert math.isclose(loss, 0, abs_tol=0.1)


class CouplingLayerTest(CouplingLayer):
    def __init__(self, *args):
        super().__init__(*args)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(8, 1),
            torch.nn.Tanh(),
            torch.nn.Linear(1, 16),
        )

    def net_forward(self, x):
        v_in = x[..., self._condition_mask]
        v_out = self.net(v_in)
        params = torch.zeros(
            x.shape[0],
            2,
            *x.shape[1:],
        )
        params.masked_scatter_(self._transform_mask, v_out)
        return params


def test_shift_and_rescale_coupling():
    prior = Prior(
        distribution=torch.distributions.Normal(
            loc=torch.zeros(LATTICE_SHAPE).unsqueeze(0),
            scale=torch.ones(LATTICE_SHAPE).unsqueeze(0),
        ),
        batch_size=100,
    )
    target = torch.distributions.Normal(
        loc=torch.full(LATTICE_SHAPE, fill_value=1.0).unsqueeze(0),
        scale=torch.full(LATTICE_SHAPE, fill_value=0.5).unsqueeze(0),
    )

    mask = torchlft.utils.make_checkerboard(LATTICE_SHAPE).unsqueeze(0)
    flow = Flow(
            CouplingLayerTest(AffineTransform(), mask),
            CouplingLayerTest(AffineTransform(), ~mask),
    )

    loss = _train_loop(prior, target, flow)

    assert math.isclose(loss, 0, abs_tol=0.1)
