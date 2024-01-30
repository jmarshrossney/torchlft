from pathlib import Path
import logging
from typing import Any, ClassVar, Self, TypeAlias

import torch

from torchlft.nflow.io import TrainingDirectory
from torchlft.utils.torch import dict_stack

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Tensor: TypeAlias = torch.Tensor

class Logger:
    """
    Collects logged metrics.
    Potentially linked to a training directory into which
    expensive things (e.g. large tensors, images) are immediately logged.
    Or just collects summary stats and aggregates them.
    """

    def __init__(self, train_dir: TrainingDirectory | None = None):
        if train_dir is None:
            logger.warning("No logging directory specified!")

        self._train_dir = train_dir
        self._steps = []
        self._metrics = []

    @property
    def log_dir(self) -> Path:
        return self._train_dir.log_dir

    def update(self, metrics: dict[str, Tensor], step: int):
        assert step not in self._steps
        if self._steps:
            assert step > max(self._steps)
        self._steps.append(step)
        self._metrics.append(metrics)

    def as_dict(self) -> dict[str, Tensor]:
        steps = torch.tensor(self._steps, dtype=torch.long)
        return dict_stack(self._metrics)

