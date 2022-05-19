from __future__ import annotations

import math
import random

import pytest
import torch

import torchlft.metrics

# This is a fairly useless test.
# TODO: throw random inputs in with hypothesis

SEED = 8967452301


@pytest.fixture
def metrics():
    random.seed(SEED)
    _ = torch.random.manual_seed(SEED)
    q = torch.distributions.Normal(0, 1)
    x = q.sample([100000])
    p = torch.distributions.Normal(0, 1.2)
    log_weights = p.log_prob(x) - q.log_prob(x)
    return torchlft.metrics.LogWeightMetrics(log_weights)


def test_metrics(metrics):
    assert math.isclose(metrics.kl_divergence, 0.0285, abs_tol=1e-4)
    assert math.isclose(metrics.acceptance, 0.8841, abs_tol=1e-4)
    assert math.isclose(
        metrics.integrated_autocorrelation, 0.7164, abs_tol=1e-4
    )
    assert math.isclose(metrics.effective_sample_size, 0.8913, abs_tol=1e-4)
    assert metrics.longest_rejection_run == 24
