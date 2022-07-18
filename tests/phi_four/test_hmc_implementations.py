from functools import partial

import torch

from torchlft.phi_four.actions import phi_four_action
from torchlft.phi_four.sample import HamiltonianMonteCarlo
from torchlft.phi_four.mcmc import hmc


def test_comparison():
    lattice_shape = [6, 6]
    beta = 0.537
    lamda = 0.5
    trajectory_length = 1.0
    steps = 10
    trajectories = 100

    cls = HamiltonianMonteCarlo(
        lattice_shape,
        trajectory_length,
        steps,
        beta=beta,
        lamda=lamda,
    )
    cls.init()

    func = partial(
        hmc,
        action=lambda phi: phi_four_action(
            phi.unsqueeze(0), beta=beta, lamda=lamda
        ).squeeze(),
        tau=trajectory_length,
        n_steps=steps,
    )

    # (1) Test that they build identical Markov chains
    phi = cls.state.clone()
    rng_state = torch.get_rng_state()

    transitions = 0
    for _ in range(trajectories):
        t = cls()
        transitions += t

    assert transitions

    torch.set_rng_state(rng_state)
    for _ in range(trajectories):
        __ = func(phi)

    assert torch.allclose(phi, cls.state)

    # (2) Sanity check that they are non-equal after another transition of
    # one chain but not the other
    transitions = 0
    while not transitions:
        t = func(phi)
        transitions += t
    assert not torch.allclose(phi, cls.state)

    # (3) Sanity check that different rng states give different results
    phi = cls.state.clone()
    for _ in range(trajectories):
        t = cls()
        transitions += t

    assert transitions

    for _ in range(trajectories):
        __ = func(phi)

    assert not torch.allclose(phi, cls.state)
