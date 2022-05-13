from __future__ import annotations

from typing import Optional

import torch

import torchlft.flows.base
import torchlft.transforms


class UnconditionalLayer(torchlft.flows.base.FlowLayer):
    """Wraps around a transformation, adding a set of learnable parameters."""

    def __init__(
        self,
        transform: torchlft.transforms.Transform,
        init_params: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self._transform = transform
        self._transform_params = torch.nn.Parameter(
            init_params or transform.identity_params
        )

    def forward(self, x: torch.Tensor, log_det_jacob: torch.Tensor):
        params = torch.stack(
            [param.expand_as(x) for param in self._transform_params.split(1)],
            dim=self._transform.params_dim,
        )
        y, ldj = self._transform(x, params)
        log_det_jacob.add_(ldj.flatten(start_dim=1).sum(dim=1))
        return y, log_det_jacob

    def inverse(self, y: torch.Tensor, log_det_jacob: torch.Tensor):
        params = torch.stack(
            [param.expand_as(y) for param in self._transform_params.split(1)],
            dim=self._transform.params_dim,
        )
        x, ldj = self._transform.inv(y, params)
        log_det_jacob.add_(ldj.flatten(start_dim=1).sum(dim=1))
        return x, log_det_jacob


class GlobalRescalingLayer(UnconditionalLayer):
    def __init__(self) -> None:
        super().__init__(transform=torchlft.transforms.Rescaling())
