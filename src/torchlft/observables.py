from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING
from math import pi as π, prod, sin

import torch
import torch.nn.functional as F

from torchlft.utils.sphere import spherical_triangle_area

if TYPE_CHECKING:
    from torchlft.typing import *


class IntegratedAutocorrelation:
    class Errors(NamedTuple):
        stat: Tensor
        bias: Tensor
        grad_stat: Tensor
        grad_bias: Tensor

    def __init__(self, observable: Tensor) -> Tensor:
        assert observable.dim() > 0

        self._sample_size = len(observable)
        self._autocorrelation = self._compute_autocorrelation(observable)

    @property
    def sample_size(self):
        return self._sample_size

    @property
    def autocorrelation(self) -> Tensor:
        return self._autocorrelation

    def estimate_errors(
        self, factor: float = 2.0
    ) -> IntegratedAutocorrelation.Errors:
        # TODO: want option to inspect the process more thoroughly
        τ_int = self.autocorrelation.cumsum(dim=-1)
        λ = factor
        N = self.sample_size
        W = torch.arange(N)

        τ_exp = (
            ((2 * τ_int - 1) / (2 * τ_int + 1)).log().reciprocal().negative()
        )
        τ_exp = τ_exp.nan_to_num().clamp(min=1e-6)

        # Statistical error (Eq. 42 in arxiv.org/pdf/hep-lat/0306017)
        ε_stat = torch.sqrt((4 / N) * (W + 1 / 2 - τ_int)) * τ_int

        # Truncation bias
        ε_bias = -τ_int * torch.exp(-W / (λ * τ_exp))

        # λ times W-derivative of the errors
        dεdW_stat = τ_exp / torch.sqrt(N * W)
        dεdW_bias = (-1 / λ) * torch.exp(-W / (λ * τ_exp))

        return self.Errors(
            stat=ε_stat,
            bias=ε_bias,
            grad_stat=dεdW_stat,
            grad_bias=dεdW_bias,
        )

    def compute(self, factor: float = 2.0) -> Tensor:
        τ_int = self.autocorrelation.cumsum(dim=-1)

        errors = self.estimate_errors(factor)
        grad_error = errors.grad_stat + errors.grad_bias

        # argmax returns first occurrence of derivative being positive, indicating
        # that the total error will increase with increasing window size
        idx = (
            torch.argmax((grad_error[..., 1:] > 0).int(), dim=-1, keepdim=True)
            + 1
        )

        return τ_int[idx]

    @staticmethod
    def _compute_autocorrelation(observable: Tensor) -> Tensor:
        assert observable.dim() > 0

        sample_size, *observable_shape = observable.shape

        # move sample dim to -1
        observable = observable.movedim(0, -1).contiguous()

        # prepare dims for conv1d
        observable = observable.view(-1, 1, 1, sample_size)

        autocovariance = []
        for o in observable:
            autocovariance.append(
                F.conv1d(
                    F.pad(o, (0, sample_size - 1)),
                    o,
                ).squeeze()
            )

        autocovariance = torch.stack(autocovariance).view(
            *observable_shape, sample_size
        )
        autocorrelation = autocovariance / autocovariance[..., 0:1]

        return autocorrelation


class _Observables:
    def __init__(
        self,
        samples: Tensor,
        func: Callable[Tensor, Tensor],
        contains_replicas: bool = False,
        n_bootstrap: int | None = None,
    ):
        if contains_replicas is True and n_bootstrap is not None:
            raise ValueError(
                "Bootstrapping only supported for a single sample"
            )

        if not contains_replicas:
            samples.unsqueeze_(dim=0)  # add dummy replica dimension

        n_replica, n_sample, lattice_L, lattice_T, *_ = samples.shape

        assert n_sample > 1

        self.n_replica = n_replica
        self.n_sample = n_sample
        self.n_bootstrap = n_bootstrap
        self.lattice_shape = (lattice_L, lattice_T)
        self.func = func

        computed = []

        # Compute func for all replica samples (may be just one)
        for sample in samples:
            computed.append(func(sample))

        # Compute func for all bootstrap samples
        if n_bootstrap is not None:
            samples.squeeze_(dim=0)
            for _ in range(n_bootstrap):
                indices = torch.randint(0, n_sample, [n_sample])
                computed.append(func(sample[indices]))

        self._computed = torch.stack(computed)


class OnePointObservables(_Observables):
    @property
    def value(self) -> Tensor:
        return reversed(torch.std_mean(self._computed, dim=0, correction=1))


class TwoPointObservables:
    """
    Observables based on two point scalar correlation function.
    """

    @property
    def correlator(self) -> Tensor:
        return self._computed

    @property
    def zero_momentum_correlator(self) -> Tensor:
        return self.correlator.sum(dim=1)

    @property
    def effective_pole_mass(self) -> Tensor:
        g_0t = self.zero_momentum_correlator
        return torch.acosh((g_0t[:, :-2] + g_0t[:, 2:]) / (2 * g_0t[:, 1:-1]))

    @property
    def susceptibility(self) -> Tensor:
        return self.correlator.sum(dim=(1, 2))

    @property
    def energy_density(self) -> Tensor:
        return (self.correlator[:, 1, 0] + self.correlator[:, 0, 1]) / 2

    @property
    def correlation_length(self) -> Tensor:
        r"""
        Estimator for correlation length based on low-momentum behaviour.

        The square of the correlation length is computed as

        .. math::

            \xi^2 = \frac{1}{4 \sin( \pi / L)} \left(
            \frac{ \tilde{G}(0, 0) }{ \mathrm{Re}\tilde{G}(2\pi/L, 0) }
            - 1 \right)

        where :math:`\tilde{G}(q, \omega)` is the Fourier transform of the
        correlator.

        Reference: :doi:`10.1103/PhysRevD.58.105007`
        """
        L, T = self.lattice_shape
        g_00 = self.susceptibility
        g_10 = self.correlator.mul(
            torch.cos(2 * π / L * torch.arange(L)).unsqueeze(dim=-1)
        ).sum(dim=(1, 2))
        xi_sq = (g_00 / g_10 - 1) / (4 * sin(π / L) ** 2)

        # NOTE: sqrt of negative entries will evaluate to nan
        # use nanmean / tensor[~tensor.isnan].std()
        return xi_sq.sqrt()
