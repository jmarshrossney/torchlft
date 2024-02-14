from typing import TypeAlias

import torch
import torch.nn as nn

Tensor: TypeAlias = torch.Tensor



def get_jacobian(transform, inputs: Tensor):
    def forward(input):
        output, *_ = transform(input.unsqueeze(0))
        output = output.squeeze(0)
        return output, output

    jac, outputs = torch.vmap(torch.func.jacrev(forward))(inputs)

    return jac, inputs, outputs


@torch.no_grad()
def get_model_jacobian(model, batch_size: int):
    inputs, _ = model.sample_base(batch_size)
    return get_jacobian(model.flow_forward, inputs)
