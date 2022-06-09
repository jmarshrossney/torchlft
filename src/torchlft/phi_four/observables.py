from typing import Optional
import math
import logging

import torch


log = logging.getLogger(__name__)


class TwoPointObservables:
    def __init__(
        self,
        sample: torch.Tensor,
        n_bootstrap_samples: int = 0,
    ) -> None:
        if sample.dim() not in (3, 4):
            raise ValueError(
                f"Expected sample dim to be 3 or 4, but got {sample.dim()}"
            )
        if sample.dim() == 4 and n_bootstrap_samples > 0:
            log.warning(
                "Concurrent sample dimension exists; bootstrapping will \
not be performed even though 'n_bootstrap_sample > 0'"
            )

        if sample.dim() == 3:
            self._samples = "bootstrap"
            self._n_samples = n_bootstrap_samples
            self._sample_size, self._lattice_L, self._lattice_T = list(
                sample.shape
            )
            self._correlator = self._compute_correlator_bootstrap(sample)

        elif sample.dim() == 4:
            self._samples = "independent"
            (
                self._n_samples,
                self._sample_size,
                self._lattice_L,
                self._lattice_T,
            ) = list(sample.shape)
            self._correlator = self._compute_correlator_independent(sample)

        # Name the dimensions of self._correlator
        self._correlator.names = ("N", "L", "T")

    def _compute_correlator(
        self, sample: torch.Tensor, resample: bool = False
    ) -> torch.Tensor:
        fweights = (
            torch.randint(0, self._sample_size, [self._sample_size]).bincount(
                minlength=self._sample_size
            )
            if resample
            else None
        )
        covariance = torch.cov(
            sample.flatten(start_dim=1).T, fweights=fweights
        )
        return torch.stack(
            [
                row.view(self._lattice_L, self._lattice_T).roll(
                    ((-i // self._lattice_T), (-i % self._lattice_L)),
                    dims=(0, 1),
                )
                for i, row in enumerate(covariance.split(1, dim=0))
            ],
            dim=0,
        ).mean(dim=0)

    def _compute_correlator_independent(
        self,
        samples: torch.Tensor,
    ) -> torch.Tensor:
        output = []
        for sample in samples.split(1, dim=0):
            output.append(self._compute_correlator(sample))
        return torch.stack(output, dim=0)

    def _compute_correlator_bootstrap(
        self, sample: torch.Tensor
    ) -> torch.Tensor:
        # First entry is original sample, without resamploing
        output = [self._compute_correlator(sample)]
        for _ in range(self._n_samples):
            output.append(self._compute_correlator(sample, resample=True))
        return torch.stack(output, dim=0)

    @property
    def samples(self) -> str:
        return self._samples

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def sample_size(self) -> int:
        return self._sample_size

    @property
    def correlator(self) -> torch.Tensor:
        return self._correlator

    @property
    def zero_momentum_correlator(self) -> torch.Tensor:
        return self._correlator.sum(dim="L")

    @property
    def effective_pole_mass(self) -> torch.Tensor:
        g_tilde_0t = self.zero_momentum_correlator
        return (
            g_tilde_0t[:, :-2]
            .add(g_tilde_0t[:, 2:])
            .div(g_tilde_0t[:, 1:-1])
            .div(2)
            .acosh()
        )

    @property
    def susceptibility(self) -> torch.Tensor:
        return self._correlator.sum(dim=("L", "T"))

    @property
    def ising_energy(self) -> torch.Tensor:
        return self._correlator[:, 1, 0].add(self._correlator[:, 0, 1]).div(2)

    @property
    def low_momentum_correlation_length(self) -> torch.Tensor:
        r"""
        Estimator for correlation length based on low-momentum behaviour.

        The square of the correlation length is computed as

        .. math::

            \xi^2 = \frac{1}{4 \sin( \pi / L)} \left(
            \frac{ \tilde{G}(0, 0) }{ \mathrm{Re}\tilde{G}(2\pi/L, 0) }
            - 1 \right)

        where :math:`\tilde{G}(q, \omega)` is the Fourier transform of the
        correlator.

        Reference: `https://doi.org/10.1103/PhysRevD.58.105007`_
        """
        g_tilde_00 = self.susceptibility
        g_tilde_10 = self._correlator.mul(
            torch.cos(
                2 * math.pi / self._lattice_L * torch.arange(self._lattice_L)
            ).unsqueeze(dim=-1)
        ).sum(dim=("L", "T"))
        xi_sq = (
            g_tilde_00.div(g_tilde_10)
            .sub(1)
            .div(4 * pow(math.sin(math.pi / self._lattice_L), 2))
        )
        # NOTE: sqrt of negative entries will evaluate to nan
        # use nanmean / tensor[~tensor.isnan].std()
        return xi_sq.sqrt()
