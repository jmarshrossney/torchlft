from dataclasses import dataclass, field

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
