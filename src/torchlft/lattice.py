from typing import TypeAlias

import torch
import torch.nn as nn

from torchlft.utils.lattice import checkerboard_mask

Tensor: TypeAlias = torch.Tensor
BoolTensor: TypeAlias = torch.BoolTensor


class Lattice2d:
    def __init__(self, L: int, T: int):
        self._dimensions = (L, T)

    @property
    def dimensions(self) -> tuple[int, int]:
        return self._dimensions


class Checkerboard2d(nn.Module):
    def __init__(self):
        super().__init__()

        self._dimensions = None
        self.register_buffer("mask", torch.BoolTensor([]), persistent=False)

    @property
    def dimensions(self) -> tuple[int, int]:
        return self._dimensions

    def update_dims(self, L: int, T: int):
        if (L, T) == self.dimensions:
            return

        self._dimensions = (L, T)

        # Refresh mask
        self.mask = checkerboard_mask(self.dimensions, device=self.mask.device)

    def forward(self, offset: int | bool):
        assert isinstance(offset, int)
        offset = bool(offset % 2)

        active_mask = ~self.mask if offset else self.mask
        return active_mask, ~active_mask
