from typing import TypeAlias

import torch
import torch.nn as nn

from torchlft.utils.lattice import checkerboard_mask

Tensor: TypeAlias = torch.Tensor
BoolTensor: TypeAlias = torch.BoolTensor


class Checkerboard2d(nn.Module):
    def __init__(self, partition_id: int):
        super().__init__()

        self._partition_id = bool(partition_id % 2)

        self.register_buffer(
            "_cached_mask", torch.BoolTensor([]), persistent=False
        )

    @property
    def partition_id(self) -> int:
        return self._partition_id

    def _forward(self, dimensions):
        assert len(dimensions) == 2
        mask = checkerboard_mask(
            dimensions,
            offset=self._partition_id,
            device=self._cached_mask.device,
        )
        self._cached_mask = mask
        return mask, ~mask

    def forward(
        self, dimensions: tuple[int, int]
    ) -> tuple[BoolTensor, BoolTensor]:
        if dimensions == self._cached_mask.shape:
            mask = self._cached_mask
            return mask, ~mask

        return self._forward(dimensions)
