from collections import OrderedDict
from copy import deepcopy
from functools import wrap
import logging
import math
from matplotlib.figure import Figure
from numbers import Real
import os
import pathlib
from types import MethodType
from typing import Sequence, Mapping

import torch
import tqdm


def log_to_tensorboard(
    logger: torch.utils.tensorboard.SummaryWriter,
    tag: str,
    value: Real | str | Sequence[Real] | Mapping[str, Real] | Figure,
    **kwargs,
) -> None:
    # If single-element tensor, unpack otherwise it will
    # (otherwise it will be treated as Sequence, not Real)
    if isinstance(value, torch.Tensor):
        try:
            value = value.item()
        except ValueError:
            pass

    if isinstance(value, Real):
        logger.add_scalar(tag, value, **kwargs)
    elif isinstance(value, str):
        logger.add_text(tag, value, **kwargs)
    elif isinstance(value, Mapping):
        logger.add_scalars(tag, value, **kwargs)
    elif isinstance(value, Sequence):
        logger.add_histogram(tag, torch.as_tensor(value), **kwargs)
    elif isinstance(value, Figure):
        logger.add_figure(tag, value, **kwargs)
    else:
        raise TypeError(f"No logging rule found for data type {type(value)}")


# TODO: either csv, json or yaml
