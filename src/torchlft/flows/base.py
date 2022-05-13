from __future__ import annotations

import torch


class FlowLayer(torch.nn.Module):
    def forward(
        self, x: torch.Tensor, ldj: torch.Tensor
    ) -> tuple[torch.Tensor]:
        raise NotImplementedError

    def inverse(
        self, x: torch.Tensor, ldj: torch.Tensor
    ) -> tuple[torch.Tensor]:
        raise NotImplementedError

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
