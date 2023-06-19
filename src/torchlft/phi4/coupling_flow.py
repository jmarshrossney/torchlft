from abc import ABCMeta, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from functools import partial
from itertools import chain
import math
from typing import ClassVar, TypeAlias

import torch
import torch.nn as nn

from torchlft import constraints as constraints
from torchlft.geometry import CheckerboardGeometry2D
from torchlft.networks import Activation, NetChoices
from torchlft.nflow import NormalizingFlow
from torchlft.transforms import (
    Translation,
    Rescaling,
    AffineTransform,
    RQSplineTransform,
)
from torchlft.typing import Any, Callable, Optional, Union, Tensor, Transform

NetFactory: TypeAlias = Callable[[int, int], Callable[Tensor, Tensor]]


"""
# TODO: look into sparse formats. Both masked and lexi in one object?
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
"""


class CouplingLayer(nn.Module, metaclass=ABCMeta):
    transform: Callable[Tensor, Transform]
    transform_n_params: int | None
    net_factory: NetFactory
    geometry: CheckerboardGeometry2D

    @abstractmethod
    def forward(
        self, ϕ: tuple[Tensor, Tensor]
    ) -> tuple[tuple[Tensor, Tensor], Tensor]:
        ...


class CouplingLayerFNN(CouplingLayer):
    def __init__(self, transform: Callable[Tensor, Transform], transform_n_params: int, net_factory: NetFactory, geometry: CheckerboardGeometry):
        super().__init__()
        n_lattice = math.prod(geometry.lattice_shape)
        assert n_lattice % 2 == 0
        half_lattice = n_lattice // 2

        self.transform = transform
        self.geometry = geometry

        self.net_a = net_factory(
            half_lattice,
            half_lattice * transform_n_params,
        )
        self.net_b = net_factory(
            half_lattice,
            half_lattice * transform_n_params,
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


class AdditiveCouplingLayer(CouplingLayerFNN):
    def __init__(self, net_factory: NetFactory, lattice_shape: tuple[int, int]):
        super().__init__(
                transform=Translation,
                transform_n_params=1,
                net_factory=net_factory,
                lattice_shape=lattice_shape,
        )


class CouplingLayerCNN(CouplingLayer):
    def __init__(self, transform: Callable[Tensor, Transform], transform_n_params: int, cnn_factory: NetFactory):
        super().__init__()
        self.net_a = self.net_factory(
            1,
            self.transform_n_params,
        )
        self.net_b = self.net_factory(
            1,
            self.transform_n_params,
        )

    def register

    def forward(
        self, ϕ: Tensor
    ) -> tuple[Tensor, Tensor]:
        ϕ_a, ϕ_b = self.geometry.partition_as_masked(ϕ)

        ϕ_b_masked = self.geometry.lexi_as_masked(ϕ_b, 1)
        θ_b = self.net_b(ϕ_b.nan_to_num())
        θ_b = θ_b[:, :, self.geometry.get_mask(0)]
        f_b = self.transform(θ_b)
        ϕ_a, ldj_a = f_b(ϕ_a)

        ϕ_a_masked = self.geometry.lexi_as_masked(ϕ_a, 0)
        θ_a = self.net_a(ϕ_a_masked.nan_to_num())
        θ_a = θ_a[:, :, self.geometry.get_mask(1)]
        f_a = self.transform(θ_a)
        ϕ_b, ldj_b = f_a(ϕ_b)

        return (ϕ_a, ϕ_b), ldj_a + ldj_b

class AdditiveLayerMixin:
    transform = Translation
    transform_n_params = 1

class AffineLayerMixin:
    transform = staticmethod(
        lambda θ: AffineTransform(*θ.tensor_split(2, dim=1))
    )
    transform_n_params = 2

class SplineLayerMixin:
    transform_n_params = 3 * n_segments - 1
    net_factory = net_factory_

    @staticmethod
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

def construct_layer(
    BaseLayer: type[CouplingLayer],
    transform: Callable[Tensor, Transform],
    transform_n_params: int,
    net_factory: NetFactory,
    ) -> type[CouplingLayer]:
    
    class Layer(BaseLayer):
        transform = staticmethod(transform)
        transform_n_params = transform_n_params
        net_factory = net_factory

    return Layer


def additive_layer(
    BaseLayer: CouplingLayer,
    net_factory: NetFactory,
) -> type[CouplingLayer]:
    net_factory_ = net_factory

    class AdditiveLayer(BaseLayer):
        transform = Translation
        transform_n_params = 1
        net_factory = net_factory_

    return AdditiveLayer


def affine_layer(
    BaseLayer: CouplingLayer,
    net_factory: NetFactory,
) -> type[CouplingLayer]:
    net_factory_ = net_factory

    class AffineLayer(BaseLayer):
        transform = staticmethod(
            lambda θ: AffineTransform(*θ.tensor_split(2, dim=1))
        )
        transform_n_params = 2
        net_factory = net_factory_

    return AffineLayer


def spline_layer(
    BaseLayer: CouplingLayer,
    net_factory: NetFactory,
    *,
    n_segments: int,
    upper_bound: float,
) -> type[CouplingLayer]:
    net_factory_ = net_factory

    class SplineLayer(BaseLayer):
        transform_n_params = 3 * n_segments - 1
        net_factory = net_factory_

        @staticmethod
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

    return SplineLayer


_TRANSFORMS = [
    additive_layer,
    affine_layer,
    spline_layer,
]

class TransformOptions(Enum):
    additive = 0
    affine = 1
    spline = 2

_BASE_LAYERS = [
    CouplingLayerFNN,
    CouplingLayerCNN,
]
_NETWORKS = [
    make_fnn,
    make_cnn,
]




@dataclass(kw_only=True)
class LayerSpec:
    transform: TransformOptions
    transform_kwargs: Optional[dict[str, Any]] = None
    net: NetOptions
    net_kwargs: dict[str, Any]
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
        net_factory,
        **spec.transform_kwargs,
    )
    if spec.net.name == "fnn":

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
