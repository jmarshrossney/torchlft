from __future__ import annotations

from collections import OrderedDict
from collections.abc import Collection, Iterator, Iterable
from copy import deepcopy
from functools import wraps
import logging
import math
from matplotlib.figure import Figure
from numbers import Real
import os
import pathlib
from types import MethodType
from typing import Union, TYPE_CHECKING

import torch
import tqdm

if TYPE_CHECKING:
    from torchlft.typing import *


def log_to_tensorboard(
    logger: torch.utils.tensorboard.SummaryWriter,
    tag: str,
    value: Real | str | Sequence[Real] | Mapping[str, Real] | Figure,
    **kwargs,
) -> None:

    # If single-element tensor, unpack otherwise it will
    # (otherwise it will be treated as Iterable, not Real)
    if isinstance(value, Tensor):
        try:
            value = value.item()
        except ValueError:
            pass

    if isinstance(value, Real):
        logger.add_scalar(tag, value, **kwargs)
    elif isinstance(value, str):
        self.logger.add_text(tag, value, **kwargs)
    elif isinstance(value, Mapping):
        self.logger.add_scalars(tag, value, **kwargs)
    elif isinstance(value, Iterable):
        self.logger.add_histogram(tag, torch.as_tensor(value), **kwargs)
    elif isinstance(value, Figure):
        self.logger.add_figure(tag, value, **kwargs)
    else:
        raise TypeError(f"No logging rule found for data type {type(value)}")


# TODO: either csv, json or yaml
