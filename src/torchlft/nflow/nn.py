from dataclasses import dataclass
from enum import StrEnum
from itertools import chain
import warnings

import torch.nn as nn


warnings.filterwarnings("ignore", message="Lazy")


class Activation(StrEnum):
    identity = "Identity"
    tanh = "Tanh"
    leaky_relu = "LeakyReLU"

    def __call__(self):
        activation_cls = getattr(nn, str(self))
        return activation_cls()


"""
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
"""


def permute_io(net: nn.Module, spatial_dims: int) -> nn.Module:
    assert spatial_dims in (1, 2), "only 1- and 2-dimensional CNN supported"
    in_perm = (0, 3, 1, 2) if spatial_dims == 2 else (0, 2, 1)
    out_perm = (0, 2, 3, 1) if spatial_dims == 2 else (0, 2, 1)

    def permute_input(module, inputs):
        (input,) = inputs
        return input.permute(in_perm)

    def permute_output(module, inputs, output):
        return output.permute(out_perm)

    # NOTE: prepend needed so that permute occurs before shape inference
    # if using LazyModule
    net.register_forward_pre_hook(permute_input, prepend=True)
    net.register_forward_hook(permute_output)

    return net


@dataclass(kw_only=True)
class ConvNet2d:
    channels: list[int]
    activation: Activation
    kernel_radius: int | list[int]

    def __post_init__(self):
        if isinstance(self.kernel_radius, int):
            self.kernel_radius = [self.kernel_radius for _ in self.channels]

    def build(self):
        conv_layers = [
            nn.LazyConv2d(
                n, kernel_size=(2 * r + 1), padding=r, padding_mode="circular"
            )
            for n, r in zip(self.channels, self.kernel_radius, strict=True)
        ]
        activations = [self.activation() for _ in conv_layers]
        layers = list(chain(*zip(conv_layers, activations)))
        net = nn.Sequential(*layers)

        return permute_io(net, 2)


@dataclass(kw_only=True)
class ConvNet1d:
    channels: list[int]
    activation: Activation
    kernel_radius: int | list[int]

    def __post_init__(self):
        if isinstance(self.kernel_radius, int):
            self.kernel_radius = [self.kernel_radius for _ in self.channels]

    def build(self):
        conv_layers = [
            nn.LazyConv1d(
                n, kernel_size=(2 * r + 1), padding=r, padding_mode="circular"
            )
            for n, r in zip(self.channels, self.kernel_radius, strict=True)
        ]
        activations = [self.activation() for _ in conv_layers]
        layers = list(chain(*zip(conv_layers, activations)))
        net = nn.Sequential(*layers)

        return permute_io(net, 1)


@dataclass(kw_only=True)
class DenseNet:
    sizes: list[int]
    activation: Activation

    def build(self):
        linear_layers = [nn.LazyLinear(n) for n in self.sizes]
        activations = [self.activation() for _ in linear_layers]
        layers = list(chain(*zip(linear_layers, activations)))
        net = nn.Sequential(*layers)
        return net


@dataclass(kw_only=True)
class PointNet:
    channels: list[int]
    activation: Activation

    def build(self):
        linear_layers = [nn.LazyLinear(n) for n in self.channels]
        activations = [self.activation() for _ in linear_layers]
        layers = list(chain(*zip(linear_layers, activations)))
        net = nn.Sequential(*layers)
        return net
