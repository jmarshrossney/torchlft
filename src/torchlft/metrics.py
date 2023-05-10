from math import exp
from random import random

import torch

from torchlft.typing import *

__all__ = [
    "metropolis_test",
    "shifted_kl_divergence",
    "effective_sample_size",
    "acceptance_rate",
    "rejection_histogram",
    "integrated_autocorrelation",
]


def metropolis_test(log_weights: Tensor) -> tuple[LongTensor, BoolTensor]:
    r"""
    Subjects a set of log-weights to the Metropolis test.

    A Markov chain is constructed by running through the sample sequentially
    and transitioning to a new state with probability

    .. math::

        A(\phi \to \phi') = \min \left\{ 1,
            \exp\left( \log w(\phi') - \log w(\phi) \right) \right\}

    Args:
        log_weights
            One-dimensional tensor containing `N` log weights

    Returns:
        A tuple containing (1) a `LongTensor` with the `N-1` indices
        corresponding to the state of the Markov chain at each step, and
        (2) a `BoolTensor` with the `N-1` accept/reject history of the
        process, with `True` corresponding to transitions.

    .. note::

        The Markov chain is initialised using the first log-weight.
    """
    log_weights = log_weights.tolist()
    current = log_weights.pop(0)

    idx = 0
    indices = []
    history = []

    for proposal in log_weights:
        # Deal with this case separately to avoid overflow
        if (proposal > current) or (
            random() < min(1, exp(proposal - current))
        ):
            current = proposal
            idx += 1
            history.append(True)
        else:
            history.append(False)

        indices.append(idx)

    indices = torch.tensor(indices, dtype=torch.long)
    history = torch.tensor(history, dtype=torch.bool)

    return indices, history


def shifted_kl_divergence(log_weights: Tensor) -> Tensor:
    r"""
    Shifted Kullbach-Leibler divergence of the sample.

    .. math::

        \tilde{D}_{KL} = \frac{1}{N} \sum_{i=1}^N - \log w(\phi_i)
    """
    return log_weights.mean(dim=-1).negative()


def effective_sample_size(log_weights: Tensor) -> Tensor:
    r"""
    Effective sample size normalised by the size of the sample.

    The effective sample size is defined by

    .. math::

        N_{eff} = \frac{
            \left[ \frac{1}{N} \sum_{i=1}^N w(\phi_i) \right]^2
            }{
            \frac{1}{N} \sum_{i=1}^N w(\phi_i)^2
        }

    The quantity returned is actually :math:`N_{eff} / N`.

    References:
        :arxiv:`2101.08176`
    """
    *_, N = log_weights.shape
    Neff = torch.exp(
        2 * torch.logsumexp(log_weights, dim=-1)
        - torch.logsumexp(2 * log_weights, dim=-1)
    )
    return Neff / N


def acceptance_rate(history: Tensor) -> Tensor:
    r"""
    Fraction of proposals accepted during Metropolis-Hastings.
    """
    return history.float().mean(dim=-1)


def rejection_histogram(history: BoolTensor) -> LongTensor:
    # Input: True = acceptance, False = rejection
    # 1 = rejection, 0 = acceptance
    history = history.logical_not().long()
    *extra_dims, length = history.shape

    # Counts no. instances of 't' consec. rej.
    histogram = torch.zeros_like(history, dtype=torch.long)

    # Tracks no. consec. rej. in current run
    this_rejection_run = torch.zeros((*extra_dims, 1), dtype=torch.long)

    step_counter = torch.arange(1, length + 1)

    for step in history.split(1, dim=-1):
        this_rejection_run += step  # add 1 for rej, 0 for acc
        this_rejection_run *= step  # mult by 0 for acc, 1 for rej

        # Add one to counter for each i <= t
        histogram += (step_counter <= this_rejection_run).long()

    return histogram


def integrated_autocorrelation(history: BoolTensor) -> Tensor:
    r"""
    Integrated autocorrelation derived from the accept/reject history.

    References:
        :arxiv:`1904.12072`
    """
    # constraints.bool.check(history)  # meh

    histogram = rejection_histogram(history)

    # Normalise histogram to make it a probability mass function
    # Normalisation = count we would have if all rejected
    *_, length = history.shape

    autocorrelation = histogram / torch.arange(length, 0, -1)

    integrated = autocorrelation.sum(dim=-1) + (1 / 2)

    return integrated
