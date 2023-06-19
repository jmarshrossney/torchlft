from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

from torchlft.typing import Tensor


class BaseAction(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

        self.register_buffer("_reference", torch.tensor([0.0]))

    @abstractmethod
    def compute(self, inputs: Tensor | tuple[Tensor, ...]) -> Tensor:
        ...

    @abstractmethod
    def gradient(
        self, inputs: Tensor | tuple[Tensor, ...]
    ) -> Tensor | tuple[Tensor, ...]:
        ...

    @abstractmethod
    def sample(
        self, sample_size: int, lattice_shape: tuple[int, ...]
    ) -> Tensor | tuple[Tensor, ...]:
        ...

    @abstractmethod
    def log_norm(self, lattice_shape: tuple[int, ...]) -> float:
        ...
