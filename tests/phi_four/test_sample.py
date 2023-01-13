from hypothesis import assume, given, settings, strategies as st
import math
import pytest
import torch

from torchlft.phi_four.sample import (
    RandomWalkMetropolis,
    Phi4HMC
)
from torchlft.sample.sampler import Sampler


@given(lattice_shape=st.lists(st.integers(2, 4), min_size=1, max_size=4))
@settings(max_examples=10, deadline=500)
def test_rw_metropolis_runs(lattice_shape):
    assume(math.prod(lattice_shape) <= 64)  # otherwise it takes too long

    algorithm = RandomWalkMetropolis(
        lattice_shape, step_size=0.1, beta=0.5, lamda=0.5
    )
    sampler = Sampler(algorithm)

    sampler.thermalise(10)
    sampler.sample(10)

    sweep = math.prod(lattice_shape)
    assert algorithm.global_step == 20 * sweep


@given(lattice_shape=st.lists(st.integers(2, 4), min_size=1, max_size=4))
@settings(max_examples=10, deadline=500)
def test_hmc_runs(lattice_shape):
    assume(math.prod(lattice_shape) <= 64)

    algorithm = Phi4HMC(
        lattice_shape, trajectory_length=1.0, steps=2, beta=0.5, lamda=0.5
    )
    sampler = Sampler(algorithm)

    sampler.thermalise(10)
    sampler.sample(10)

    assert algorithm.global_step == 20


@given(lattice_shape=st.lists(st.integers(2, 4), min_size=1, max_size=4))
@settings(max_examples=10, deadline=500)
def test_hmc_mass_matrix_runs(lattice_shape):
    assume(math.prod(lattice_shape) <= 64)

    n_lattice = math.prod(lattice_shape)
    _x = torch.rand(n_lattice, n_lattice).tril()
    mass_matrix = torch.mm(_x, _x.T).add(torch.ones(n_lattice).diag())

    algorithm = Phi4HMC(
        lattice_shape,
        trajectory_length=1.0,
        steps=2,
        mass_matrix=mass_matrix,
        beta=0.5,
        lamda=0.5,
    )
    sampler = Sampler(algorithm)

    sampler.thermalise(10)
    sampler.sample(10)

    assert algorithm.global_step == 20
