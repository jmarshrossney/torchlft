from __future__ import annotations

from typing import Optional

import torch


class Compose(torch.nn.Sequential):
    def forward(
        self, x: torch.Tensor, log_det_jacob: Optional[torch.Tensor] = None
    ):
        log_det_jacob = log_det_jacob or torch.zeros(x.shape[0]).type_as(x)
        for module in self:
            x, log_det_jacob = module.forward(x, log_det_jacob)
        return x, log_det_jacob

    def inverse(
        self, x: torch.Tensor, log_det_jacob: Optional[torch.Tensor] = None
    ):
        log_det_jacob = log_det_jacob or torch.zeros(x.shape[0]).type_as(x)
        for module in self:
            x, log_det_jacob = module.inverse(x, log_det_jacob)
        return x, log_det_jacob
