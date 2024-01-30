from typing import TypeAlias

import torch
import torch.nn as nn

Tensor: TypeAlias = torch.Tensor


class Composition(nn.Module):
    def __init__(self, *layers: nn.Module):
        super().__init__()
        self.register_module("layers", nn.ModuleList(layers))

    def __iter__(self):
        return iter(self.layers)

    def forward(self, inputs: Tensor | tuple[Tensor, ...]):
        ldj_total = 0.0

        for layer in self:
            outputs, ldj = layer(inputs)
            ldj_total += ldj

            inputs = outputs

        return outputs, ldj_total
