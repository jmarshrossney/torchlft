from __future__ import annotations

import torch


class FlowLayer(torch.nn.Module):
    def forward(
        self, x: torch.Tensor, log_prob: torch.Tensor
    ) -> tuple[torch.Tensor]:
        raise NotImplementedError

    def inverse(
        self, x: torch.Tensor, log_prob: torch.Tensor
    ) -> tuple[torch.Tensor]:
        raise NotImplementedError

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True


class Flow(torch.nn.Sequential):
    def forward(self, x: torch.Tensor, log_prob: torch.Tensor):
        for layer in self:
            x, log_prob = layer.forward(x, log_prob)
        return x, log_prob

    def inverse(self, y: torch.Tensor, log_prob: torch.Tensor):
        for layer in self:
            y, log_prob = layer.inverse(y, log_prob)
        return y, log_prob
