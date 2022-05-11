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
        log_det_jacob = log_det_jacob or torch.zeros(x.shape[0]).type_as(y)
        for module in self:
            y, log_det_jacob = module.inverse(x, log_det_jacob)
        return y, log_det_jacob


class _Old_UnconditionalTransform(torch.nn.Module):
    """Wraps around a transformation, adding a set of learnable parameters."""

    def __init__(self, transform):
        super().__init__()
        self.transform = transform
        self.params = torch.nn.Parameter(
            transform.identity_params.view(1, 1, -1)
        )

    def forward(self, inputs, log_det_jacob):
        outputs, log_det_jacob_this = self.transform(
            inputs, self.params.expand(inputs.shape[0], 1, -1)
        )
        log_det_jacob.add_(log_det_jacob_this)
        return outputs, log_det_jacob

    def inverse(self, inputs, log_det_jacob):
        outputs, log_det_jacob_this = self.transform.inverse(
            inputs, self.params.expand(inputs.shape[0], 1, -1)
        )
        log_det_jacob.add_(log_det_jacob_this)
        return outputs, log_det_jacob


class _UnconditionalTransform:
    def __init__(
        self, transform: torchlft, params: dict, trainable: list[str]
    ) -> None:
        if trainable:
            self._params = {
                param: torch.nn.Parameter(tensor)
                for param, tensor in params.items()
            }
        else:
            self._params = params

    def forward(
        self, x: torch.Tensor, log_det_jacob: torch.Tensor
    ) -> tuple[torch.Tensor]:
        y, ldj = F.translation(x, **self._params, **self._kwargs)
        log_det_jacob.add_(ldj)
        return y, log_det_jacob

    def inverse(
        self, y: torch.Tensor, log_det_jacob: torch.Tensor
    ) -> tuple[torch.Tensor]:
        x, ldj = F.transforms.inv_translation(y, self.shift)
        log_det_jacob.add_(ldj)
        return x, log_det_jacob
