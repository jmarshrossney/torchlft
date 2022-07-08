from copy import deepcopy

import pytest
import torch
from torchlft.common.mcmc import OneShotSampler


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
def oneshot_simple(simple_generator):
    return OneShotSampler(simple_generator)


def test_oneshot_update(oneshot_simple):
    sampler = oneshot_simple
    assert sampler.global_step == 0
    assert sampler.transitions == 0
    sampler_init_state = sampler.get_state()

    state = sampler.init()

    steps = 0
    transitions = 0
    while transitions in (0, steps):
        prev_state = deepcopy(state)
        transitioned = sampler.update(state)

        if transitioned:
            assert state != prev_state
        else:
            assert state == prev_state

        transitions += int(bool(transitioned))

    assert sampler.global_step == 0
    assert sampler.transitions == 0
    assert sampler.get_state() == sampler_init_state


def test_oneshot_forward(oneshot_simple):
    sampler = oneshot_simple
    assert sampler.global_step == 0
    assert sampler.transitions == 0

    while sampler.transitions in (0, sampler.global_step):
        (output,) = sampler().values()

    assert sampler.global_step > 0
    assert sampler.transitions > 0
    assert sampler.get_state() == output

    output["config"] = None
    assert sampler._state["config"] is not None
