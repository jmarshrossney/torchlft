from hypothesis import given, settings, strategies as st
import math
import pytest
import torch

from torchlft.phi_four.sample import RandomWalkMetropolis
from torchlft.sample.sampler import Sampler


@given(lattice_shape=st.lists(st.integers(2, 4), min_size=1, max_size=4))
@settings(max_examples=10, deadline=400)
def test_rw_metropolis_runs(lattice_shape):
    model = RandomWalkMetropolis(
        lattice_shape, step_size=0.1, beta=0.5, lamda=0.5
    )
    sampler = Sampler()

    sampler.thermalise(model, 10)
    sampler.sample(model, 10)

    sweep = math.prod(lattice_shape)
    assert model.global_step == 20 * sweep
