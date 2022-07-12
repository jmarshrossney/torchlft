from copy import deepcopy

import pytest
import torch

from torchlft.sample.algorithms import MCMCReweighting
from torchlft.sample.sampler import Sampler


@pytest.fixture
def simple_generator():
    def generator():
        prior = torch.distributions.Normal(0, 1)
        target = torch.distributions.Normal(0, 0.8)
        while True:
            sample = prior.sample()
            log_prob_prior = prior.log_prob(sample)
            log_prob_target = target.log_prob(sample)
            log_weight = log_prob_target - log_prob_prior
            yield sample, log_weight

    return generator()


@pytest.fixture
def algorithm(simple_generator):
    return MCMCReweighting(simple_generator)


@pytest.fixture
def sampler(algorithm):
    return Sampler(algorithm)


def test_algorithm(algorithm):
    assert algorithm.global_step is None
    assert algorithm.transitions is None
    assert algorithm.state is None

    algorithm.init()

    assert algorithm.global_step == 0
    assert algorithm.transitions == 0
    assert isinstance(algorithm.state, torch.Tensor)

    assert "state" in algorithm.state_dict()

    algorithm.init()


def test_thermalise(sampler):
    sampler.thermalise(10)
    assert sampler.algorithm.global_step == 10


def test_forward_hooks(algorithm):
    assert algorithm.global_step is None
    assert algorithm.transitions is None

    algorithm.init()

    assert algorithm.global_step == 0
    assert algorithm.transitions == 0

    step = 0
    state = deepcopy(algorithm.state)
    while algorithm.state == state:
        algorithm()
        step += 1

    assert algorithm.global_step == step
    assert algorithm.transitions == 1


def test_sample(sampler):
    out = sampler.sample(10)
    assert len(out) == 10
