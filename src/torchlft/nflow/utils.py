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


def get_jacobian(transform, inputs: Tensor):
    def forward(input):
        output, _ = transform(input.unsqueeze(0))
        output = output.squeeze(0)
        return output, output

    jac, outputs = torch.vmap(torch.func.jacrev(forward))(inputs)

    return jac, inputs, outputs


def get_model_jacobian(model, batch_size: int):
    inputs, _ = model.sample_base(batch_size)
    return get_jacobian(model.flow_forward, inputs)
