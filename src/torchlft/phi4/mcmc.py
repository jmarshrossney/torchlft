from __future__ import annotations

import os
import pathlib
from collections.abc import Iterator
from math import exp
import math
from random import random
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.utils.tensorboard as tensorboard
import tqdm

from torchlft.actions import (
    phi_four_action,
    phi_four_action_local,
)
from torchlft.sample import SamplingAlgorithm
from torchlft.utils.lattice import build_neighbour_list


class RandomWalkMetropolis(SamplingAlgorithm):
    def __init__(
        self,
        lattice_shape: torch.Size,
        step_size: float,
        **couplings: dict[str, float],
    ) -> None:
        super().__init__()
        self.lattice_shape = lattice_shape
        self.step_size = step_size
        self.couplings = couplings

        self.lattice_size = math.prod(lattice_shape)
        self.neighbour_list = build_neighbour_list(lattice_shape)

    @property
    def sweep_length(self) -> int:
        return self.lattice_size

    def init(self) -> None:
        self.state = torch.empty(self.lattice_shape).normal_(0, 1)

        # This is just a view of the original state
        self.flattened_state = self.state.view(-1)

    def forward(self) -> bool:
        site_idx = torch.randint(0, self.lattice_size, [1]).item()
        neighbour_idxs = self.neighbour_list[site_idx]

        phi_old, *neighbours = self.flattened_state[
            [site_idx, *neighbour_idxs]
        ]
        phi_new = phi_old + torch.randn(1).item() * self.step_size

        old_action = phi_four_action_local(
            phi_old, neighbours, **self.couplings
        )
        new_action = phi_four_action_local(
            phi_new, neighbours, **self.couplings
        )

        if metropolis_test(old_action - new_action):
            self.flattened_state[site_idx] = phi_new
            return True
        else:
            return False
