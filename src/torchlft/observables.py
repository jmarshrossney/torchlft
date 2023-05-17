from __future__ import annotations

from typing import TYPE_CHECKING
from math import pi as π, prod, sin

import torch
import torch.nn.functional as F

from torchlft.abc import Field
from torchlft.geometry import spherical_triangle_area

if TYPE_CHECKING:
    from torchlft.typing import *


def autocorrelation(observable: Iterable) -> Tensor:
    # TODO: vectorise
    observable = torch.as_tensor(observable)
    assert observable.dim() == 1
    n = observable.shape[0]

    observable = observable.view(1, 1, -1)
    autocovariance = torch.nn.functional.conv1d(
        F.pad(observable, (0, n - 1)),
        observable,
    ).squeeze()

    return autocovariance.div(autocovariance[0])


def magnetisation_sq(sample: CanonicalClassicalSpinField) -> Tensor:
    return sample.tensor.pow(2).flatten(start_dim=1).mean(dim=1)


def topological_charge(sample: CanonicalClassicalSpinField) -> Tensor:
    assert len(sample.lattice_shape) == 2
    assert sample.element_shape == (3,)

    s1 = sample.data
    s2 = s1.roll(-1, 1)
    s3 = s2.roll(-1, 2)
    s4 = s3.roll(+1, 1)

    area_enclosed = (
        spherical_triangle_area(s1, s2, s3)
        + spherical_triangle_area(s1, s3, s4)
    ).sum(dim=(1, 2))

    return area_enclosed / (4 * π)


def two_point_correlator_scalar(configs: CanonicalScalarField) -> Tensor:
    assert (
        len(configs.lattice_shape) == 2
    ), "Only 2d lattices supported at this time"

    correlator_lexi = torch.corrcoef(
        configs.tensor.flatten(start_dim=1).transpose(),
    )

    # Takes a volume average of 2-dimensional shifts
    # For each lattice site (row), restores the geometry by representing
    # the row as a 2d array where the axis correspond to displacements
    # in the two lattice directions. Then takes a volume average by
    # averaging over all rows (lattice sites)
    # TODO: generalise to lattice dimensions 1, 2, 3, 4 (not just 2)
    L, T = configs.lattice_shape
    return torch.stack(
        [
            row.view(L, T).roll(
                ((-i // T), (-i % L)),
                dims=(0, 1),
            )
            for i, row in enumerate(correlator_lexi)
        ],
        dim=0,
    ).mean(dim=0)


def spin_spin_correlator(sample: CanonicalClassicalSpinField) -> Tensor:
    assert (
        len(sample.lattice_shape) == 2
    ), "Only 2d lattices supported at this time"

    sample = sample.to_canonical()

    sample_lexi = sample.tensor.flatten(start_dim=1, end_dim=-2)
    correlator_lexi = torch.einsum(
        "bia,bja->ij", sample_lexi, sample_lexi
    ) / len(sample_lexi)

    L, T = configs.lattice_shape
    return torch.stack(
        [
            row.view(L, T).roll(
                ((-i // T), (-i % L)),
                dims=(0, 1),
            )
            for i, row in enumerate(correlator_lexi.split(1, dim=0))
        ],
        dim=0,
    ).mean(dim=0)


class _Observables:
    def __init__(
        self,
        *samples: Field,
        func: Callable[[Field, LongTensor | None], Tensor],
        n_bootstrap: int | None = None,
    ):
        n_independent = len(samples)

        if n_bootstrap is not None and n_independent > 1:
            raise ValueError

        for attr in ("__class__", "__len__", "lattice_shape", "element_shape"):
            assert len(set([getattr(sample, attr) for sample in samples])) == 1

        n_batch = len(samples[0])
        assert n_batch > 1

        lattice_shape = samples[0].lattice_shape
        assert len(lattice_shape) != 2, "Currently only 2d lattices supported"

        self.n_bootstrap = n_bootstrap
        self.n_batch = n_batch
        self.lattice_shape = lattice_shape
        self.func = func

        computed = []
        for sample in samples:
            computed.append(func(sample))

        if n_bootstrap is not None:
            (sample,) = samples
            for _ in range(n_bootstrap):
                indices = torch.randint(0, n_batch, [n_batch])
                computed.append(func(sample[indices]))

        self._computed = torch.stack(computed, dim=0)


class OnePointObservables(_Observables):
    def __init__(
        *samples: Field,
        one_point_func: Callable[Field, Tensor],
        n_bootstrap: int | None = None,
    ):
        super().__init__(
            *samples,
            func=one_point_func,
            n_bootstrap=n_bootstrap,
        )

    @property
    def value(self) -> Tensor:
        return reversed(torch.std_mean(self._computed, dim=0, correction=1))


class TwoPointObservables:
    """
    Observables based on two point scalar correlation function.
    """

    def __init__(
        self,
        *samples: Field,
        two_point_func: Callable[Field, Tensor],
        n_bootstrap: int | None = None,
    ):
        super().__init__(
            *samples,
            func=two_point_func,
            n_bootstrap=n_bootstrap,
        )
        self._correlator = self._computed

    @property
    def correlator(self) -> Tensor:
        return self._corr

    @property
    def zero_momentum_correlator(self) -> Tensor:
        return self._correlator.sum(dim=1)

    @property
    def effective_pole_mass(self) -> Tensor:
        g_0t = self.zero_momentum_correlator
        return torch.acosh((g_0t[:, :-2] + g_0t[:, 2:]) / (2 * g_0t[:, 1:-1]))

    @property
    def susceptibility(self) -> Tensor:
        return self._correlator.sum(dim=(1, 2))

    @property
    def energy_density(self) -> Tensor:
        return (self._correlator[:, 1, 0] + self._correlator[:, 0, 1]) / 2

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
        g_10 = self._correlator.mul(
            torch.cos(2 * π / L * torch.arange(L)).unsqueeze(dim=-1)
        ).sum(dim=(1, 2))
        xi_sq = (g_00 / g_10 - 1) / (4 * sin(π / L) ** 2)

        # NOTE: sqrt of negative entries will evaluate to nan
        # use nanmean / tensor[~tensor.isnan].std()
        return xi_sq.sqrt()
