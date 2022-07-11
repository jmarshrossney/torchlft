from collections.abc import Iterable
from itertools import chain, cycle

from jsonargparse.typing import PositiveInt, PositiveFloat
import torch

from torchlft.common.mcmc import Sampler, OneShotSampler


class MetropolisSampler(Sampler):
    def __init__(
        self,
        lattice_shape: Iterable[PositiveInt],
        step_size: PositiveFloat,
        action: Callable[torch.Tensor, torch.Tensor],
    ) -> None:
        super().__init__(
            lattice_shape=lattice_shape, step_size=step_size, action=action
        )

        self._subvol_mask = torch.nn.functional.pad(
            torch.ones([3 for _ in lattice_shape]),
            (i for i in chain.from_iterable(zip(cycle(0), lattice_shape))),
            mode="constant",
            value=0,
        ).roll((-1 for _ in lattice_shape), tuple(range(len(lattice_shape))))


    def init(self) -> dict:
        return {"config": torch.empty(self.lattice_shape).normal_()}

    def update(self, state: dict) -> bool:
V
        site = torch.randi
