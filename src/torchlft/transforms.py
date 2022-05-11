from __future__ import annotations

import torch

import torchlft.functional as F
import torchlft.utils


class Transform:
    def __call__(
        self, x: torch.Tensor, **params: torch.Tensor
    ) -> tuple[torch.Tensor]:
        raise NotImplementedError

    def inv(
        self, x: torch.Tensor, **params: torch.Tensor
    ) -> tuple[torch.Tensor]:
        raise NotImplementedError


class Translation(Transform):
    def __call__(
        self, x: torch.Tensor, shift: torch.Tensor
    ) -> tuple[torch.Tensor]:
        return F.translation(x, shift)

    def inv(self, y: torch.Tensor, shift: torch.Tensor) -> tuple[torch.Tensor]:
        return F.inv_translation(y, shift)


class AffineTransform(Transform):
    def __call__(
        self, x: torch.Tensor, log_scale: torch.Tensor, shift: torch.Tensor
    ) -> tuple[torch.Tensor]:
        return F.translation(x, log_scale, shift)

    def inv(
        self, y: torch.Tensor, log_scale: torch.Tensor, shift: torch.Tensor
    ) -> tuple[torch.Tensor]:
        return F.transforms.inv_translation(y, log_scale, shift)


class RQSplineTransform(Transform):
    def __init__(
        self,
        interval: tuple[float],
        domain: str,
    ) -> None:
        self.interval = interval
        self.domain = domain

    def __call__(
        self,
        x: torch.Tensor,
        widths: torch.Tensor,
        heights: torch.Tensor,
        derivs: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        spline_params = torchlft.utils.build_rq_spline(
            widths, heights, derivs, self.interval, self.domain
        )
        return F.transforms.rq_spline_transform(x, *spline_params)

    def inv(
        self,
        y: torch.Tensor,
        widths: torch.Tensor,
        heights: torch.Tensor,
        derivs: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        spline_params = torchlft.utils.build_rq_spline(
            widths, heights, derivs, self.interval, self.domain
        )
        return F.transforms.inv_rq_spline_transform(y, *spline_params)
