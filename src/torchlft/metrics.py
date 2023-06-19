from math import exp
from random import random

import torch

from torchlft.typing import Tensor, BoolTensor, LongTensor

__all__ = [
    "metropolis_test",
    "LogWeightMetrics",
]


def metropolis_test(
    log_weights: Tensor,
) -> tuple[LongTensor, BoolTensor, LongTensor]:
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

    rejection_histogram = [0 for _ in range(len(log_weights) - 1)]
    rejection_counter = 0

    for proposal in log_weights:
        # Deal with this case separately to avoid overflow
        if (proposal > current) or (
            random() < min(1, exp(proposal - current))
        ):
            current = proposal
            idx += 1
            history.append(True)

            for i, j in enumerate(range(rejection_counter, 0, -1)):
                rejection_histogram[i] += j

            rejection_counter = 0
        else:
            history.append(False)
            rejection_counter += 1

        indices.append(idx)

    indices = torch.tensor(indices, dtype=torch.long)
    history = torch.tensor(history, dtype=torch.bool)
    rejection_histogram = torch.tensor(rejection_histogram, dtype=torch.long)

    return indices, history, rejection_histogram


class LogWeightMetrics:
    def __init__(self, log_weights: Tensor) -> Tensor:
        assert log_weights.dim() == 1
        self.log_weights = log_weights
        self.indices, self.history, self.rejection_histogram = metropolis_test(
            log_weights
        )

    @property
    def mean_log_weight(self) -> Tensor:
        return self.log_weights.mean()

    @property
    def variance_log_weight(self) -> Tensor:
        return self.log_weights.var()

    @property
    def effective_sample_size(self) -> Tensor:
        N = len(self.log_weights)
        Neff = torch.exp(
            2 * torch.logsumexp(self.log_weights, dim=0)
            - torch.logsumexp(2 * self.log_weights, dim=0)
        )
        return Neff / N

    @property
    def acceptance_rate(self) -> Tensor:
        return self.history.float().mean()

    @property
    def integrated_autocorrelation(self) -> Tensor:
        chain_length = len(self.history)

        autocorrelation = self.rejection_histogram / torch.arange(
            chain_length, 0, -1
        )

        integrated = autocorrelation.sum() + (1 / 2)

        return integrated

    @property
    def most_consecutive_rejections(self) -> Tensor:
        return self.rejection_histogram.nonzero().max().float()

    def asdict(self) -> dict[str, Tensor]:
        return {
            "mean_log_weight": self.mean_log_weight,
            "variance_log_weight": self.variance_log_weight,
            "effective_sample_size": self.effective_sample_size,
            "acceptance_rate": self.acceptance_rate,
            "integrated_autocorrelation": self.integrated_autocorrelation,
            "most_consecutive_rejections": self.most_consecutive_rejections,
        }


"""
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
"""
