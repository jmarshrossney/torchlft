from __future__ import annotations

from itertools import chain
from typing import Union

from jsonargparse.typing import PositiveInt, restricted_number_type
import torch

# So the help isn't flooded with all available nn.Modules
# NOTE: torch master branch has __all__ attribute. When that is merged replace
# explicit list with torch.nn.modules.activations.__all__
_ACTIVATIONS = tuple(
    getattr(torch.nn, a)
    for a in [
        "ELU",
        "Hardshrink",
        "Hardsigmoid",
        "Hardtanh",
        "Hardswish",
        "LeakyReLU",
        "LogSigmoid",
        "PReLU",
        "ReLU",
        "ReLU6",
        "RReLU",
        "SELU",
        "CELU",
        "GELU",
        "Sigmoid",
        "SiLU",
        "Mish",
        "Softplus",
        "Softshrink",
        "Softsign",
        "Tanh",
        "Tanhshrink",
        "Threshold",
        "GLU",
        "Identity",
    ]
)
Activation = Union[_ACTIVATIONS]


def make_fnn(
    size_in: PositiveInt,
    size_out: PositiveInt,
    hidden_shape: list[PositiveInt],
    activation: Activation,
    final_activation: Activation = torch.nn.Identity(),
    bias: bool = True,
) -> torch.nn.Sequential:
    layers = [
        torch.nn.Linear(f_in, f_out, bias=bias)
        for f_in, f_out in zip(
            [size_in, *hidden_shape], [*hidden_shape, size_out]
        )
    ]
    activations = [activation for _ in hidden_shape] + [final_activation]
    return torch.nn.Sequential(*list(chain(*zip(layers, activations))))


def make_cnn(
    dim: ConvDim,
    in_channels: PositiveInt,
    out_channels: PositiveInt,
    hidden_shape: list[PositiveInt],
    kernel_size: PositiveInt,
    activation: Activation,
    final_activation: Activation = torch.nn.Identity(),
    circular: bool = False,
    conv_kwargs: dict = {},
) -> torch.nn.Sequential:
    Conv = {
        1: torch.nn.Conv1d,
        2: torch.nn.Conv2d,
        3: torch.nn.Conv3d,
    }.__getitem__(dim)
    if circular:
        conv_kwargs.update(padding=kernel_size // 2, padding_mode="circular")
    layers = [
        Conv(c_in, c_out, kernel_size, **conv_kwargs)
        for c_in, c_out in zip(
            [in_channels, *hidden_shape], [*hidden_shape, out_channels]
        )
    ]
    activations = [activation for _ in hidden_shape] + [final_activation]
    return torch.nn.Sequential(*list(chain(*zip(layers, activations))))
