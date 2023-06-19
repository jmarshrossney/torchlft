from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from torchlft.utils import StrEnum


class Activation(StrEnum):
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


class NetFactory(ABC):
    @abstractmethod
    def __call__(self, size_in: int, size_out: int, **context: dict[str, Any]) -> nn.Sequential:
        ...

@dataclass(kw_only=True)
class FnnFactory(NetFactory):
    hidden_shape: list[int, ...]
    activation: Activation
    bias: bool
    final_activation: Optional[Activation] = None

    def __post_init__(self) -> None:
        self.activation = getattr(nn, str(self.activation))
        self.final_activation = (
            getattr(nn, str(self.final_activation))
            if self.final_activation is not None
            else nn.Identity
        )

    def __call__(self, size_in: int, size_out: int) -> nn.Sequential:
        layers = [
            nn.Linear(f_in, f_out, bias=bias)
            for f_in, f_out in zip(
                [size_in, *self.hidden_shape], [*self.hidden_shape, size_out]
            )
        ]
        activations = [self.activation() for _ in self.hidden_shape] + [
            self.final_activation()
        ]
        return nn.Sequential(*list(chain(*zip(layers, activations))))


@dataclass(kw_only=True)
class CnnFactory(NetFactory):
    hidden_shape: list[int, ...]
    activation: Activation
    bias: bool
    kernel_radius: int
    final_activation: Optional[Activation] = None
    conv_kwargs: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.activation = getattr(nn, str(self.activation))
        self.final_activation = (
            getattr(nn, str(self.final_activation))
            if self.final_activation is not None
            else nn.Identity
        )

    def __call__(self, size_in: int, size_out: int) -> nn.Sequential:
        kernel_size = self.kernel_radius * 2 + 1
        kwargs = conv_kwargs | {
            "padding": kernel_size // 2,
            "padding_mode": "circular",
        }
        layers = [
            nn.Conv2d(c_in, c_out, kernel_size, **kwargs)
            for c_in, c_out in zip(
                [channels_in, *self.hidden_shape],
                [*self.hidden_shape, channels_out],
            )
        ]
        activations = [self.activation() for _ in self.hidden_shape] + [
            self.final_activation()
        ]
        return nn.Sequential(*list(chain(*zip(layers, activations))))


IMPLEMENTED_NETWORKS = {
    "fnn": FnnFactory,
    "cnn": CnnFactory,
}


class NetChoices(StrEnum):
    fnn = 0
    cnn = 1


def get_net_factory(choice: NetChoices):
    return IMPLEMENTED_NETWORKS[str(choice)]
