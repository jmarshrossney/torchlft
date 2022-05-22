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

    def forward(self, x: torch.Tensor, log_prob: torch.Tensor):
        params = torch.stack(
            [param.expand_as(x) for param in self._transform_params.split(1)],
            dim=self._transform.params_dim,
        )
        y, ldj = self._transform(x, params)
        log_prob.sub_(ldj)
        return y, log_prob

    def inverse(self, y: torch.Tensor, log_prob: torch.Tensor):
        params = torch.stack(
            [param.expand_as(y) for param in self._transform_params.split(1)],
            dim=self._transform.params_dim,
        )
        x, ldj = self._transform.inv(y, params)
        log_prob.sub_(ldj)
        return x, log_prob


class GlobalRescalingLayer(UnconditionalLayer):
    def __init__(self) -> None:
        super().__init__(transform=torchlft.transforms.Rescaling())
