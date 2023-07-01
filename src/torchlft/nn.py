from ABC import ABCMeta, abstractmethod
import torch
import torch.nn as nn

Tensor = torch.Tensor

class TransformModule(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def forward(self, *args, **kwargs) -> Transform:
        ...


class FNNTransformModule(TransformModule):

    def forward(self, inputs: Tensor) -> Transform:
        params = self.net(inputs)
        return self.transform(params)
