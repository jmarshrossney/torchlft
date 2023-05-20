from __future__ import annotations

import torch
import torch.nn as nn

from torchlft.abc import Geometry

Tensor = torch.Tensor


class NormalizingFlow(nn.Module):
    def __init__(self, geometry: Geometry, layers):
        super().__init__()
        self.geometry = geometry
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs: Tensor) -> tuple[tuple[Tensor, ...], Tensor]:
        partitions = self.geometry.partition(inputs)

        ldj_total = torch.zeros(len(inputs), device=inputs.device)

        for layer in self.layers:
            partitions, ldj = layer(partitions)
            ldj_total += ldj

            self.on_layer(partitions, ldj_total)

        outputs = self.geometry.restore(partitions)

        return outputs, ldj_total

    def on_layer(self, partitions: tuple[Tensor, ...], ldj: Tensor) -> None:
        pass
