from __future__ import annotations

from typing import TYPE_CHECKING
from math import pi as π, prod, sin

import torch
import torch.nn.functional as F

from torchlft.abc import Field
from torchlft.geometry import spherical_triangle_area

if TYPE_CHECKING:
    from torchlft.typing import *


def magnetisation_sq(sample: CanonicalClassicalSpinField) -> Tensor:
    return sample.tensor.pow(2).flatten(start_dim=1).mean(dim=1)


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
