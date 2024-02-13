from abc import ABC, abstractmethod
from collections import OrderedDict
import logging
from typing import Any, TypeAlias

import torch

from torchlft.nflow.io import TrainingDirectory
from torchlft.utils.torch import dict_stack

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Tensor: TypeAlias = torch.Tensor


class Logger(ABC):
    def __init__(self):
        self._train_dir = None

    @property
    def train_dir(self) -> TrainingDirectory | None:
        return self._train_dir

    @train_dir.setter
    def train_dir(self, td: TrainingDirectory) -> None:
        if self._train_dir is not None:
            raise Exception(
                "There is already a training directory associated with this logger!"
            )

        assert isinstance(td, TrainingDirectory)
        assert td.root.is_dir()

        td.log_dir.mkdir(parents=False, exist_ok=True)

        self._train_dir = td

    @abstractmethod
    def update(self, data: Any, step: int, **extra_context) -> None:
        ...


class MonotoneIntegerDict(OrderedDict):
    def __setitem__(self, key, value):
        assert isinstance(key, int)
        if bool(self):
            assert key > max(self.keys())
        super().__setitem__(key, value)


class DefaultLogger(Logger):
    """
    Collects logged data and serves them as a dict of tensors.
    Potentially linked to a training directory into which
    expensive things (e.g. large tensors, images) are immediately logged.
    """

    def __init__(self):
        super().__init__()
        self._data = MonotoneIntegerDict()

    def update(self, data: dict[str, Tensor], step: int):
        self.data[step] = data

    @property
    def data(self) -> dict[int, dict[str, Tensor]]:
        return self._data

    def get_data(self) -> dict[str, Tensor]:
        steps, data = zip(*self._data.items())
        steps = torch.tensor(steps, dtype=torch.long)
        data = dict_stack(data)

        return {"steps": steps} | data
