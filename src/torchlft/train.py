from abc import ABC, abstractmethod
import logging
from typing import Any, TypeAlias

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from tqdm import trange

from torchlft.model import Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Tensor: TypeAlias = torch.Tensor


class Trainer(ABC):
    @abstractmethod
    def train(self, model: Model, context: Any = None):  # -> TrainingMetrics:
        ...


# TODO: disable progress bar option


class ReverseKLTrainer(Trainer):
    def __init__(
        self,
        *,
        n_steps: int,
        batch_size: int,
        log_metrics: bool = True,
        log_interval: int = 100,
        log_batch_size: int = 1024,
        log_n_batches: int = 10,
        clip_grad_norm: float | None = None,
        pbar_interval: int = 10,
        print_model_summary: bool = True,
    ):
        self.n_steps = n_steps
        self.batch_size = batch_size
        # self.init_lr = init_lr
        self.log_metrics = log_metrics
        self.log_interval = log_interval
        self.log_batch_size = log_batch_size
        self.log_n_batches = log_n_batches
        self.clip_grad_norm = clip_grad_norm
        self.pbar_interval = pbar_interval
        self.print_model_summary = print_model_summary

    # @abstractmethod
    def configure_optimizers(
        self, model: Model, context: Any = None
    ) -> tuple[Optimizer, Scheduler]:
        ...

    def training_step(self, model: Model, step: int) -> Tensor:
        fields, actions = model(self.batch_size)

        log_weights = actions.pushforward - actions.target
        rev_kl = log_weights.mean().negative()

        return rev_kl

    def metrics_step(self, model: Model, step: int):
        #metrics = TrainingMetrics()

        for _ in range(self.metrics_n_batches):
            fields, actions = model(self.metrics_batch_size)
            #metrics.update(fields, actions)

        #computed_metrics = metrics.compute()
        # training_log.update(computed_metrics, step=step)
        #print(computed_metrics)

    def train(self, model: Model):  # -> ComputedTrainingMetrics:
        # TODO: make configurable
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.001
        )  # self.init_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_steps
        )

        _ = model(1)
        if self.print_model_summary:
            print(model)  # log?

        # training_log = TrainingLog()

        with trange(self.n_steps + 1, desc="Training") as pbar:
            pbar_stats = {}
            for step in pbar:
                loss = self.training_step(model, step)

                if step % self.pbar_interval == 0:
                    pbar_stats |= {"loss": f"{loss.float():.5f}"}
                    pbar.set_postfix(pbar_stats)

                optimizer.zero_grad()
                loss.backward()

                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.clip_grad_norm
                    )

                optimizer.step()
                scheduler.step()

                if step % self.metrics_interval == 0:
                    model.eval()
                    metrics = self.metrics_step(model, step)
                    # pbar_stats |= {
                    #    "ess": f"{computed_metrics.ess.mean.float():.3f}"
                    # }
                    # pbar.set_postfix(pbar_stats)
                    model.train()

        return  # training_log
