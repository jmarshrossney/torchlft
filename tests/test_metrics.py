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
    log_weights = torch.rand(10000)
    return torchlft.metrics.LogWeightMetrics(log_weights)


def test_metrics(metrics):
    assert math.isclose(metrics.kl_divergence, 0.49898, abs_tol=1e-4)
    assert math.isclose(metrics.acceptance, 0.83268, abs_tol=1e-4)
    assert math.isclose(
        metrics.integrated_autocorrelation, 0.72413, abs_tol=1e-4
    )
    assert math.isclose(metrics.effective_sample_size, 0.92328, abs_tol=1e-4)
    assert metrics.longest_rejection_run == 7
