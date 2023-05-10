from abc import ABCMeta

import pytorch_lightning as pl
import torchmetrics

from torchlft.typing import *
from torchlft.metrics import (
    shifted_kl_divergence,
    acceptance_rate,
    effective_sample_size,
    rejection_histogram,
    integrated_autocorrelation,
)

__all__ = [
    "ShiftedKLDivergence",
    "EffectiveSampleSize",
    "AcceptanceRate",
    "IntegratedAutocorrelation",
    "LogWeightsMetrics",
]


class _LogWeightsMetric(torchmetrics.Metric, metaclass=ABCMeta):
    r"""
    Abstract base class for metrics based on log-weights.

    'Log-weight' refers to the logarithm of the statistical weight for
    generated configuration, :math:`\phi`, defined up to an additive
    constant by

    .. math::

        \log w(\phi) = \log p(\phi) - \log q(\phi)

    where :math:`p` and :math:`q` are the target and model probability
    densities, respectively.
    """

    is_differentiable = True
    full_state_update = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "log_weights",
            default=list(),
            dist_reduce_fx=None,
        )

    def update(self, log_weights: Tensor) -> None:
        if log_weights.dim() != 1:
            raise ValueError(
                f"wrong number of dimensions: expected 1, got {log_weights.dim()}"
            )
        self.log_weights.append(log_weights)

    def compute(self) -> Tensor:
        raise NotImplementedError


class _LogWeightsMetricMCMC(LogWeightMetric):
    """
    Base class for metrics arising from a Metropolis-Hastings simulation.

    .. seealso: :py:func:`torchlft.metrics.metropolis_test`
    """

    is_differentiable = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "history",
            default=list(),
            dist_reduce_fx=None,
        )

    def update(self, log_weights: Tensor) -> None:
        super().update(log_weights)

        _, history = metropolis_test(log_weights)
        self.history.append(history)


class ShiftedKLDivergence(_LogWeightsMetric):
    """Shifted Kullbach-Leibler divergence of the sample.

    See also: :py:func:`torchlft.metrics.shifted_kl_divergence`
    """

    higher_is_better = False

    def compute(self) -> Tensor:
        return shifted_kl_divergence(torch.stack(self.log_weights))


class EffectiveSampleSize(_LogWeightsMetric):
    """Effective sample size normalised by the size of the sample.

    See also: :py:func:torchlft.metrics.effective_sample_size
    """

    higher_is_better = True

    def compute(self) -> Tensor:
        return effective_sample_size(torch.stack(self.log_weights))


class AcceptanceRate(_LogWeightsMetricMCMC):
    """
    Fraction of proposals accepted during Metropolis-Hastings.

    See also: :py:func:`torchlft.metrics.acceptance_rate`
    """

    higher_is_better = True

    def compute(self) -> Tensor:
        return acceptance_rate(torch.stack(self.history))


class IntegratedAutocorrelation(_LogWeightsMetricMCMC):
    """
    Integrated autocorrelation derived from the accept/reject history.

    See Also: :py:func:`torchlft.metrics.integrated_autocorrelation`
    """

    higher_is_better = False

    def compute(self) -> Tensor:
        return integrated_autocorrelation(torch.stack(self.history))


class LogWeightsMetrics(torchmetrics.MetricCollection):
    """
    Collection of metrics computed using log-weights.
    """

    def __init__(self) -> None:
        metrics = [
            ShiftedKLDivergence(),
            EffectiveSampleSize(),
            AcceptanceRate(),
            IntegratedAutocorrelation(),
        ]
        compute_groups = [
            ["ShiftedKLDivergence", "EffectiveSampleSize"],
            [
                "AcceptanceRate",
                "IntegratedAutocorrelation",
            ],
        ]
        super().__init__(metrics=metrics, compute_groups=compute_groups)
