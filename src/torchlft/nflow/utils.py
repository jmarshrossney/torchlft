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


def get_jacobian(model, input: Tensor | None = None):
    if input is None:
        input, _ = model.base(1)
        input = input[0]

    def forward(input):
        output, _ = model.flow_forward(input.unsqueeze(0))
        output = output.squeeze(0)
        return output, output

    jac, output = torch.func.jacrev(forward)(input)

    return jac, input, output
