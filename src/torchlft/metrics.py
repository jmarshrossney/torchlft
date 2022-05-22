from __future__ import annotations

import math
import random

import torch

Tensor = torch.Tensor


class LogWeightMetrics:
    r"""Lightweight metrics computed using the log-weights of a model output.

    'Log-weight' refers to the logarithm of the statistical weight of a
    generated configuration, :math:`\Psi`, defined

    .. math::

        \log w(\Psi) = \log p(\Psi) - \log q(\Psi)

    where :math:`p(\Psi)` and :math:`q(\Psi)` are the target and model
    probability densities, respectively.

    Parameters
    ----------
    log_weights
        One-dimensional tensor of log-weights
    """

    def __init__(self, log_weights: Tensor) -> None:
        assert (
            log_weights.dim() == 1
        ), "log_weights should be a 1-dimensional tensor"
        self._log_weights = log_weights
        self._run_metropolis_hastings()

    def _run_metropolis_hastings(self) -> None:
        log_weights = self._log_weights.tolist()
        curr_log_weight = log_weights.pop(0)
        history = []

        for prop_log_weight in log_weights:
            # Deal with this case separately to avoid overflow
            if prop_log_weight > curr_log_weight:
                curr_log_weight = prop_log_weight
                history.append(1)
            elif random.random() < min(
                1, math.exp(prop_log_weight - curr_log_weight)
            ):
                curr_log_weight = prop_log_weight
                history.append(1)
            else:
                history.append(0)

        self._history = history

    @property
    def kl_divergence(self) -> float:
        r"""Kullbach-Leibler divergence of the sample.

        .. math::

            D_{KL} = \frac{1}{N} \sum_{\{\Phi\}} - \log w(\Psi)
        """
        return float(self._log_weights.mean().neg())

    @property
    def acceptance(self) -> float:
        r"""Fraction of proposals accepted during Metropolis-Hastings.

        Given target probability density :math:`p(\phi)`, candidate
        configurations :math:`\{ \phi \sim \tilde{p}(\phi) \}` are
        accepted with a probability

        .. math::

            A(\phi \to \phi^\prime) = \min \left( 1,
            \frac{\tilde{p}(\phi)}{p(\phi)}
            \frac{p(\phi^\prime)}{\tilde{p}(\phi^\prime)} \right) \, .

        Generally speaking, the acceptance rate will be larger if the overlap
        between the model and the target densities is larger.
        """
        return sum(self._history) / len(self._history)

    @property
    def longest_rejection_run(self) -> int:
        r"""Largest number of consecutive rejections in the sampling phase."""
        n = 0
        n_max = 0
        for step in self._history:
            if not step:
                n += 1
                if n > n_max:
                    n_max = n
            else:
                n = 0
        return n_max

    @property
    def integrated_autocorrelation(self) -> float:
        r"""Integrated autocorrelation derived from the accept/reject history.

        In the limit :math:`t \to \infty` the autocorrelation function can be
        re-interpreted as the probability of failing to transition over a given
        number of Markov chain steps.

        .. math::

            \frac{\Gamma(t)}{\Gamma(0)} =
            \Pr(t \text{ consecutive rejections} )

        An estimate of this probability is obtained by simply summing the
        number of occurrences of a :math:`t` consecutive rejections, and
        normalizing by the largest number of occurrences that are possible;
        that is, by :math:`T - t + 1` where :math:`T` is the total number
        of Markov chain steps.

        References: https://arxiv.org/abs/1904.12072
        """
        tally = [0 for _ in range(len(self._history))]  # first element is t=1
        n_rej = 0

        for step in self._history:
            if step:  # candidate accepted
                if n_rej > 0:
                    for t in range(n_rej):
                        tally[t] += n_rej - t
                n_rej = 0
            else:  # candidate rejected
                n_rej += 1

        for t in range(n_rej):  # catch last run
            tally[t] += n_rej - t

        # Normalize
        autocorr = [
            a / b for a, b in zip(tally, range(len(self._history), 0, -1))
        ]

        return 0.5 + sum(autocorr)

    @property
    def effective_sample_size(self) -> float:
        r"""Effective sample size normalised by the size of the sample.

        .. math::

            N_{eff} = \frac{
                \left[ \frac{1}{N} \sum_{\{\Phi\}} w(\Phi) \right]^2
                }{
                \frac{1}{N} \sum_{\{\Phi\}} w(\Phi)^2
            }

        References
        ----------
        https://arxiv.org/abs/2101.08176
        """
        ess = torch.exp(
            self._log_weights.logsumexp(dim=0).mul(2)
            - self._log_weights.mul(2).logsumexp(dim=0)
        )
        return float(ess.div(len(self._log_weights)))
