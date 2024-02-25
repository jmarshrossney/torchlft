from abc import ABCMeta, abstractmethod
from typing import TypeAlias

import torch
import torch.nn as nn

from torchlft.nflow.model import Model

Tensor: TypeAlias = torch.Tensor
Tensors: TypeAlias = tuple[Tensor, ...]

class Action(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def forward(self, inputs: Tensor | Tensors) -> Tensor:
        ...

    @abstractmethod
    def grad(self, inputs: Tensor | Tensors) -> Tensor | Tensors:
        ...


class PullbackAction(Action):
    def __init__(self, model: Model):
        super().__init__()
        self.register_module("model", model)

    @torch.no_grad()
    def forward(self, inputs: Tensor | Tensors) -> Tensor:
        outputs, log_det_jacobian = self.model.flow_forward(inputs)
        pullback = self.model.compute_target(outputs) - log_det_jacobian
        return pullback

    @torch.enable_grad()
    def grad(self, inputs: Tensor | Tensors) -> Tensor | Tensors:
        return self.model.grad_pullback(inputs)



