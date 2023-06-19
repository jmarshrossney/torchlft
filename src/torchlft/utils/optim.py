from dataclasses import dataclass, field
import functools
import types

import torch
import torch.nn as nn

from torchlft.typing import Optional, Optimizer, Scheduler


def freeze(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def unfreeze(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True


@dataclass
class OptimizerConfig:
    optimizer: str
    optimizer_init: dict = field(default_factory=dict)
    scheduler: Optional[str] = None
    scheduler_init: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.optimizer = getattr(torch.optim, self.optimizer)
        if self.scheduler is not None:
            self.scheduler = getattr(torch.optim.lr_scheduler, self.scheduler)

    def init(self, module: nn.Module) -> tuple[Optimizer, Scheduler | None]:
        optimizer = self.optimizer(module.parameters(), **self.optimizer_init)
        scheduler = (
            self.scheduler(optimizer, **self.scheduler_init)
            if self.scheduler is not None
            else None
        )
        return optimizer, scheduler


@dataclass
class OptimizerConfig:
    """
    Dataclass representing a single optimizer with optional lr scheduler.

    This class provides, via the :meth:`add_to` method, an alternative
    way to configure an optimizer and lr scheduler, as opposed to
    defining ``configure_optimizers`` in the ``LightningModule`` itself.

    Args:
        optimizer:
            The optimizer class
        optimizer_init:
            Keyword args to instantiate optimizer
        scheduler:
            The lr scheduler class
        scheduler_init:
            Keyword args to instantiate scheduler
        scheduler_extra_config:
            Extra scheduler config
        submodule:
            Optionally specify a submodule whose ``parameters()``
            will be passed to the optimizer.

    Example:

        >>> optimizer_config = OptimizerConfig(
                "Adam",
                {"lr": 0.001},
                "CosineAnnealingLR",
                {"T_max": 1000},
            )
        >>> # MyModel does not override configure_optimizers
        >>> model = MyModel(...)
        >>> optimizer_config.add_to(model)
    """

    optimizer: Union[str, type[Optimizer]]
    optimizer_init: dict = field(default_factory=dict)
    scheduler: Optional[
        Union[
            str,
            type[Scheduler],  # TODO: invalid type hint
        ]
    ] = None
    scheduler_init: dict = field(default_factory=dict)
    scheduler_extra_config: dict = field(default_factory=dict)
    submodule: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.optimizer, str):
            self.optimizer = getattr(torch.optim, self.optimizer)
        if isinstance(self.scheduler, str):
            self.scheduler = getattr(torch.optim.lr_scheduler, self.scheduler)

    def init(self, module: nn.Module) -> tuple[Optimizer, Scheduler | None]:
        optimizer = self.optimizer(module.parameters(), **self.optimizer_init)
        scheduler = (
            self.scheduler(optimizer, **self.scheduler_init)
            if self.scheduler is not None
            else None
        )
        return optimizer, scheduler

    @staticmethod
    def configure_optimizers(
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Scheduler | None,
        scheduler_extra_config: dict,
    ):
        """
        Simple function used to override ``configure_optimizers``.
        """
        return (
            optimizer
            if scheduler is None
            else {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler}
                | scheduler_extra_config,
            }
        )

    def add_to(self, model: nn.Module) -> None:
        """
        Add the optimizer and scheduler to an existing ``LightningModule``.
        """
        module = getattr(model, self.submodule) if self.submodule else model
        optimizer = self.optimizer(module.parameters(), **self.optimizer_init)
        scheduler = (
            self.scheduler(optimizer, **self.scheduler_init)
            if self.scheduler is not None
            else None
        )

        configure_optimizers = functools.partial(
            self.configure_optimizers,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_extra_config=self.scheduler_extra_config,
        )

        # Adds __wrapped__ attribute to partial fn, required for
        # PyTorch Lightning to regard configure_optimizers as overridden
        # (see pytorch_lightning.utilities.model_helpers.is_overridden)
        functools.update_wrapper(
            configure_optimizers, self.configure_optimizers
        )

        model.configure_optimizers = types.MethodType(
            configure_optimizers, model
        )
