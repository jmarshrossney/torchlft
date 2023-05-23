from enum import Enum
from itertools import chain
from typing import Union

from jsonargparse.typing import PositiveInt, restricted_number_type
import torch
import torch.nn as nn


class Activation(Enum):
    Identity = 0
    ELU = 1
    Hardshrink = 2
    Hardsigmoid = 3
    Hardtanh = 4
    Hardswish = 5
    LeakyReLU = 6
    LogSigmoid = 7
    PReLU = 8
    ReLU = 9
    ReLU6 = 10
    RReLU = 11
    SELU = 12
    CELU = 13
    GELU = 14
    Sigmoid = 15
    SiLU = 16
    Mish = 17
    Softplus = 18
    Softshrink = 19
    Softsign = 20
    Tanh = 21
    Tanhshrink = 22
    Threshold = 23
    GLU = 24


def make_fnn(
    size_in: int,
    size_out: int,
    *,
    hidden_shape: list[int],
    activation: Activation | str,
    final_activation: Activation | str | None,
    bias: bool,
) -> nn.Sequential:
    activation = getattr(
        nn,
        activation.name if isinstance(activation, Activation) else activation,
    )
    final_activation = (
        getattr(
            nn,
            final_activation.name
            if isinstance(final_activation, Activation)
            else final_activation,
        )
        if final_activation is not None
        else nn.Identity
    )
    layers = [
        nn.Linear(f_in, f_out, bias=bias)
        for f_in, f_out in zip(
            [size_in, *hidden_shape], [*hidden_shape, size_out]
        )
    ]
    activations = [activation() for _ in hidden_shape] + [final_activation()]
    return nn.Sequential(*list(chain(*zip(layers, activations))))


def make_cnn(
    channels_in: int,
    channels_out: int,
    *,
    dim: int,
    hidden_shape: list[int],
    kernel_size: int,
    activation: Activation,
    final_activation: Activation | None,
    circular: bool,
    **kwargs,
) -> nn.Sequential:
    Conv = {
        1: nn.Conv1d,
        2: nn.Conv2d,
        3: nn.Conv3d,
    }.__getitem__(dim)
    if circular:
        kwargs.update(padding=kernel_size // 2, padding_mode="circular")
    layers = [
        Conv(c_in, c_out, kernel_size, **kwargs)
        for c_in, c_out in zip(
            [channels_in, *hidden_shape], [*hidden_shape, channels_out]
        )
    ]
    activation = getattr(
        nn,
        activation.name if isinstance(activation, Activation) else activation,
    )
    final_activation = (
        getattr(
            nn,
            final_activation.name
            if isinstance(final_activation, Activation)
            else final_activation,
        )
        if final_activation is not None
        else nn.Identity
    )
    activations = [activation() for _ in hidden_shape] + [final_activation()]
    return nn.Sequential(*list(chain(*zip(layers, activations))))
