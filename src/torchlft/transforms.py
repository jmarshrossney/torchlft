from __future__ import annotations

from typing import ClassVar

import torch

import torchlft.functional.transforms as F
import torchlft.utils


class Transform:
    domain: ClassVar[str]

    def __call__(
        self, x: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor]:
        raise NotImplementedError

    def inv(
        self, x: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor]:
        raise NotImplementedError

    @property
    def n_params(self) -> int:
        raise NotImplementedError

    @property
    def params_dim(self) -> int:
        raise NotImplementedError

    def identity_params(self) -> torch.Tensor:
        raise NotImplementedError


class Translation(Transform):
    domain = "reals"

    def __call__(
        self, x: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor]:
        # NOTE: prefer view_as over squeeze since latter fails silently
        return F.translation(x, shift=params.view_as(x))

    def inv(
        self, y: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor]:
        return F.inv_translation(y, shift=params.view_as(y))

    @property
    def n_params(self) -> int:
        return 1

    @property
    def params_dim(self) -> int:
        return 1  # it doesn't matter

    @property
    def identity_params(self) -> torch.Tensor:
        return torch.Tensor([0])


class Rescaling(Transform):
    domain = "reals"

    def __call__(
        self, x: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor]:
        return F.rescaling(x, log_scale=params.view_as(x))

    def inv(
        self, y: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor]:
        return F.inv_rescaling(y, log_scale=params.view_as(y))

    @property
    def n_params(self) -> int:
        return 1

    @property
    def params_dim(self) -> int:
        return 1  # it doesn't matter

    @property
    def identity_params(self) -> torch.Tensor:
        return torch.Tensor([0])

    def get_identity_params(self, data_shape: torch.Size) -> torch.Tensor:
        return torch.zeros(data_shape).unsqueeze(self.params_dim)


class AffineTransform(Transform):
    domain = "reals"

    def __init__(self, params_dim: int = 1):
        self._params_dim = params_dim

    def __call__(
        self, x: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor]:
        log_scale, shift = [
            p.view_as(x) for p in params.split(1, dim=self._params_dim)
        ]
        return F.affine_transform(x, log_scale=log_scale, shift=shift)

    def inv(
        self, y: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor]:
        log_scale, shift = [
            p.view_as(y) for p in params.split(1, dim=self._params_dim)
        ]
        return F.inv_affine_transform(y, log_scale=log_scale, shift=shift)

    @property
    def n_params(self) -> int:
        return 2

    @property
    def params_dim(self) -> int:
        return self._params_dim

    @property
    def identity_params(self) -> torch.Tensor:
        return torch.Tensor([0, 0])


class RQSplineTransform(Transform):
    domain = "interval"

    def __init__(
        self,
        n_segments: int,
        interval: tuple[float],
        params_dim: int = 1,
    ) -> None:
        self._n_segments = n_segments
        self._interval = interval
        self._params_dim = params_dim

    def __call__(
        self, x: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor]:
        widths, heights, derivs = params.split(
            (self._n_segments, self._n_segments, self._n_knots),
            dim=self._params_dim,
        )
        spline_params = torchlft.utils.build_rq_spline(
            widths, heights, derivs, self._interval, self.domain
        )
        return F.rq_spline_transform(x, *spline_params)

    def inv(
        self, y: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor]:
        widths, heights, derivs = params.split(
            (self._n_segments, self._n_segments, self._n_knots),
            dim=self._params_dim,
        )
        spline_params = torchlft.utils.build_rq_spline(
            widths, heights, derivs, self._interval, self.domain
        )
        return F.inv_rq_spline_transform(y, *spline_params)

    @property
    def _n_knots(self) -> int:
        return self._n_segments + 1

    @property
    def n_params(self) -> int:
        return 2 * self._n_segments + self.n_knots

    @property
    def params_dim(self) -> int:
        return self._params_dim

    @property
    def identity_params(self) -> torch.Tensor:
        return torch.cat(
            (
                torch.full(
                    size=(2 * self._n_segments,),
                    fill_value=1 / self._n_segments,
                ),
                (torch.ones(self._n_knots).exp() - 1).log(),
            ),
            dim=0,
        )


class RQSplineTransformLinearTails(RQSplineTransform):
    domain = "reals"

    @property
    def _n_knots(self) -> int:
        return self._n_segments - 1


class RQSplineTransformCircular(RQSplineTransform):
    domain = "circle"

    @property
    def _n_knots(self) -> int:
        return self._n_segments
