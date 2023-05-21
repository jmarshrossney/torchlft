# from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from functools import partial
from itertools import chain
import math
from typing import (
    Any,
    Callable,
    ClassVar,
    Optional,
    TypeAlias,
    TYPE_CHECKING,
    Union,
)

import torch
import torch.nn as nn

from torchlft import constraints as constraints
from torchlft.fields import CanonicalScalarField
from torchlft.networks import make_fnn, make_cnn, Activation
from torchlft.transforms import (
    Translation,
    Rescaling,
    AffineTransform,
    RQSplineTransform,
)
from torchlft.utils.lattice import make_checkerboard

from torchlft.typing import Transform

from torchlft.theories.phi4.flow_base import NormalizingFlow

if TYPE_CHECKING:
    from torchlft.typing import *

Tensor: TypeAlias = torch.Tensor
NetFactory: TypeAlias = Callable[[int, int], Callable[Tensor, Tensor]]


class Geometry(nn.Module):
    def __init__(self, lattice_shape: tuple[int, int]):
        super().__init__()
        assert len(lattice_shape) == 2
        assert [L % 2 == 0 for L in lattice_shape]

        self._lattice_shape = tuple(lattice_shape)
        self.register_buffer("mask", make_checkerboard(lattice_shape))

    @property
    def lattice_shape(self) -> tuple[int, int]:
        return self._lattice_shape

    def partition(self, ϕ: Tensor) -> tuple[Tensor, Tensor]:
        ϕ_a, ϕ_b = ϕ[:, self.mask], ϕ[:, ~self.mask]
        return ϕ_a, ϕ_b

    def restore(self, ϕ: tuple[Tensor, Tensor]) -> Tensor:
        ϕ_a, ϕ_b = ϕ
        ϕ = torch.empty(
            (ϕ_a.shape[0], *self.lattice_shape),
            device=ϕ_a.device,
            dtype=ϕ_a.dtype,
        )
        ϕ[:, self.mask] = ϕ_a
        ϕ[:, ~self.mask] = ϕ_b
        return ϕ


class CouplingLayer(nn.Module, metaclass=ABCMeta):
    transform: Callable[Tensor, Transform]
    transform_n_params: int | None
    geometry: Geometry
    net_factory: NetFactory

    @abstractmethod
    def forward(
        self, ϕ: tuple[Tensor, Tensor]
    ) -> tuple[tuple[Tensor, Tensor], Tensor]:
        ...


class CouplingLayerFNN(CouplingLayer):
    def __init__(self):
        super().__init__()
        n_lattice = math.prod(self.geometry.lattice_shape)
        assert n_lattice % 2 == 0
        half_lattice = n_lattice // 2

        self.net_a = self.net_factory(
            half_lattice,
            half_lattice * self.transform_n_params,
        )
        self.net_b = self.net_factory(
            half_lattice,
            half_lattice * self.transform_n_params,
        )

    def forward(
        self, ϕ: tuple[Tensor, Tensor]
    ) -> tuple[tuple[Tensor, Tensor], Tensor]:
        ϕ_a, ϕ_b = ϕ

        θ_b = self.net_b(ϕ_b)
        f_b = self.transform(θ_b)
        ϕ_a, ldj_a = f_b(ϕ_a)

        θ_a = self.net_a(ϕ_a)
        f_a = self.transform(θ_a)
        ϕ_b, ldj_b = f_a(ϕ_b)

        return (ϕ_a, ϕ_b), ldj_a + ldj_b


class CouplingLayerCNN(CouplingLayer):
    def __init__(self):
        super().__init__()
        self.net_a = self.net_factory(
            1,
            self.transform_n_params,
        )
        self.net_b = self.net_factory(
            1,
            self.transform_n_params,
        )

    def forward(
        self, ϕ: tuple[Tensor, Tensor]
    ) -> tuple[tuple[Tensor, Tensor], Tensor]:
        ϕ_a, ϕ_b = ϕ
        mask = self.geometry.mask

        ϕ_b_masked = torch.zeros(
            ϕ_b.shape[0], 1, *self.geometry.lattice_shape, device=ϕ_b.device
        )
        ϕ_b_masked[:, ~mask] = ϕ_b
        θ_b = self.net_b(ϕ_b_masked)[:, :, mask]
        f_b = self.transform(θ_b)
        ϕ_a, ldj_a = f_b(ϕ_a)

        ϕ_a_masked = torch.zeros(
            ϕ_a.shape[0], 1, *self.geometry.lattice_shape, device=ϕ_a.device
        )
        ϕ_a_masked[:, mask] = ϕ_a
        θ_a = self.net_a(ϕ_a_masked)[:, :, ~mask]
        f_a = self.transform(θ_a)
        ϕ_b, ldj_b = f_a(ϕ_b)

        return (ϕ_a, ϕ_b), ldj_a + ldj_b


class GlobalRescaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self, ϕ: tuple[Tensor, Tensor]
    ) -> tuple[tuple[Tensor, Tensor], Tensor]:
        ϕ_a, ϕ_b = ϕ
        f = Rescaling(self.log_scale)
        ϕ_a, ldj_a = f(ϕ_a)
        ϕ_b, ldj_b = f(ϕ_b)
        return (ϕ_a, ϕ_b), ldj_a + ldj_b


# TODO: Net is a mixture of conv and dense layers: inputs like conv,
# outputs like dense

# TODO: Net takes additional context inputs


def additive_layer(
    BaseLayer: CouplingLayer,
    lattice_shape: tuple[int, int],
    net_factory: NetFactory,
) -> type[CouplingLayer]:
    net_factory_ = net_factory

    class AdditiveLayer(BaseLayer):
        transform = Translation
        transform_n_params = 1
        geometry = Geometry(lattice_shape)
        net_factory = net_factory_

    return AdditiveLayer


def affine_layer(
    BaseLayer: CouplingLayer,
    lattice_shape: tuple[int, int],
    net_factory: NetFactory,
) -> type[CouplingLayer]:
    net_factory_ = net_factory

    class AffineLayer(BaseLayer):
        transform = staticmethod(
            lambda θ: AffineTransform(*θ.tensor_split(2, dim=1))
        )
        transform_n_params = 2
        geometry = Geometry(lattice_shape)
        net_factory = net_factory_

    return AffineLayer


def spline_layer(
    BaseLayer: CouplingLayer,
    lattice_shape: tuple[int, int],
    net_factory: NetFactory,
    *,
    n_segments: int,
    upper_bound: float,
) -> type[CouplingLayer]:
    def transform(θ: Tensor) -> RQSplineTransform:
        w, h, d = θ.unflatten(1, (-1, 3 * n_segments - 1)).split(
            [n_segments, n_segments, n_segments - 1], dim=2
        )
        return RQSplineTransform(
            w,
            h,
            d,
            lower_bound=-upper_bound,
            upper_bound=upper_bound,
            bounded=False,
            periodic=False,
            min_slope=1e-3,
        )

    net_factory_ = net_factory

    class SplineLayer(BaseLayer):
        transform = transform
        transform_n_params = 3 * n_segments - 1
        geometry = Geometry(lattice_shape)
        net_factory = net_factory_

    return SplineLayer


_TRANSFORMS = [
    additive_layer,
    affine_layer,
    spline_layer,
]

_NETWORKS = [
    make_fnn,
    make_cnn,
]
_LAYERS = [
    CouplingLayerFNN,
    CouplingLayerCNN,
]


class _TransformOptions(Enum):
    additive = 0
    affine = 1
    spline = 2


class _NetOptions(Enum):
    fnn = 0
    cnn = 1


@dataclass(kw_only=True)
class LayerSpec:
    transform: _TransformOptions
    transform_kwargs: Optional[dict[str, Any]] = None
    net: _NetOptions
    net_hidden_shape: list[int, ...]
    net_bias: bool
    net_activation: Activation = "Tanh"
    net_final_activation: Optional[Activation] = None
    cnn_kernel_radius: Optional[int] = None
    compose: int = 1


def _make_layers(lattice_shape, spec: LayerSpec) -> list[CouplingLayer]:
    net_kwargs = dict(
        hidden_shape=spec.net_hidden_shape,
        bias=spec.net_bias,
        activation=spec.net_activation,
        final_activation=spec.net_final_activation,
    )
    if spec.net.name == "cnn":
        net_kwargs.update(kernel_size=spec.cnn_kernel_radius * 2 + 1)

    net_factory = partial(_NETWORKS[spec.net.value], **net_kwargs)

    layer_ = _TRANSFORMS[spec.transform.value]
    layer = layer_(
        _LAYERS[spec.net.value],
        lattice_shape,
        net_factory,
    )
    return [layer() for _ in range(spec.compose)]


def make_flow(
    lattice_shape: list[int, int], layers: list[LayerSpec]
) -> NormalizingFlow:
    geometry = Geometry(lattice_shape)
    layers = list(
        chain.from_iterable(
            [_make_layers(lattice_shape, spec) for spec in layers]
        )
    )
    layers.append(GlobalRescaling())
    return NormalizingFlow(geometry, layers)
