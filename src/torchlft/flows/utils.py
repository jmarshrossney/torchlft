from __future__ import annotations

from typing import Optional

import torch

import torchlft.transforms


class Compose(torch.nn.Sequential):
    def forward(
        self, x: torch.Tensor, log_det_jacob: Optional[torch.Tensor] = None
    ):
        log_det_jacob = log_det_jacob or torch.zeros(x.shape[0]).type_as(x)
        for module in self:
            x, log_det_jacob = module.forward(x, log_det_jacob)
        return x, log_det_jacob

    def inverse(
        self, y: torch.Tensor, log_det_jacob: Optional[torch.Tensor] = None
    ):
        log_det_jacob = log_det_jacob or torch.zeros(y.shape[0]).type_as(y)
        for module in self:
            y, log_det_jacob = module.inverse(y, log_det_jacob)
        return y, log_det_jacob


class UnconditionalTransform(torch.nn.Module):
    """Wraps around a transformation, adding a set of learnable parameters."""

    def __init__(
        self,
        transform: torchlft.transforms.Transform,
        init_params: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.transform = transform
        self.params = torch.nn.Parameter(
            init_params or transform.identity_params
        )

    def forward(self, x, log_det_jacob):
        params = torch.stack(
            [param.expand_as(x) for param in self.params.split(1)],
            dim=self.transform.params_dim,
        )
        y, ldj = self.transform(x, params)
        log_det_jacob.add_(ldj)
        return y, log_det_jacob

    def inverse(self, y, log_det_jacob):
        params = torch.stack(
            [param.expand_as(y) for param in self.params.split(1)],
            dim=self.transform.params_dim,
        )
        x, ldj = self.transform.inverse(y, params)
        log_det_jacob.add_(ldj)
        return x, log_det_jacob

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
