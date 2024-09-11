from abc import ABC, abstractmethod
import logging
from typing import Any, TypeAlias

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from tqdm import tqdm, trange

from torchlft.nflow.model import Model
from torchlft.nflow.metrics import LogWeightMetrics
from torchlft.nflow.logging import Logger, DefaultLogger

logging.basicConfig(level=logging.INFO)
logger_ = logging.getLogger(__name__)

Tensor: TypeAlias = torch.Tensor


class Trainer(ABC):
    @abstractmethod
    def train(self, model: Model, context: Any = None):  # -> TrainingMetrics:
        ...


# TODO: disable progress bar option
class ReverseKLTrainer(Trainer):
    logger = None

    def __init__(
        self,
        *,
        n_steps: int,
        batch_size: int,
        init_lr: float = 0.001,
        log_interval: int = 100,
        log_batch_size: int = 1024,
        log_n_batches: int = 10,
        test_batch_size: int = 1024,
        test_n_batches: int = 10,
        clip_grad_norm: float | None = None,
        print_model_summary: bool = True,
        progress_bar: bool = True,
        display_metrics: bool = True,
        progress_bar_interval: int = 10,
    ):
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.log_interval = log_interval
        self.log_batch_size = log_batch_size
        self.log_n_batches = log_n_batches
        self.test_batch_size = test_batch_size
        self.test_n_batches = test_n_batches
        self.clip_grad_norm = clip_grad_norm
        self.print_model_summary = print_model_summary
        self.enable_progress_bar = progress_bar
        self.enable_display_metrics = display_metrics
        self.progress_bar_interval = progress_bar_interval

        self.display_metrics = {}

    # @abstractmethod
    def configure_optimizers(
        self, model: Model, context: Any = None
    ) -> tuple[Optimizer, Scheduler]: ...

    def training_step(self, model: Model, step: int) -> Tensor:
        fields, actions = model(self.batch_size)

        log_weights = actions.pushforward - actions.target
        rev_kl = log_weights.mean().negative()

        return rev_kl

    def compute_log_weight_metrics(self, model: Model):
        metrics = LogWeightMetrics()

        for _ in range(self.log_n_batches):
            fields, actions = model(self.log_batch_size)
            log_weights = actions.pushforward - actions.target
            metrics.update(log_weights)

        computed_metrics = metrics.compute()
        return computed_metrics

    def logging_step(self, model: Model, step: int):
        computed_metrics = self.compute_log_weight_metrics(model)

        self.logger.update(computed_metrics, step=step)

        return {k: float(v.mean()) for k, v in computed_metrics.items()}

    def train(self, model: Model, logger: Logger | None = None):
        if logger is None:
            logger_.warning("No logger provided - using default logger.")
            logger = DefaultLogger()
        self.logger = logger

        # TODO: make configurable
        #optimizer = torch.optim.Adam(model.parameters(), lr=self.init_lr)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.init_lr)
        #scheduler = torch.optim.lr_scheduler.MultiplicativeLR
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_steps
        )

        _ = model(1)
        if self.print_model_summary:
            print(model)  # log?
            print(f"Trainable parameters: {model.parameter_count}")

        with trange(
            self.n_steps + 1,
            desc="Training",
            disable=not self.enable_progress_bar,
        ) as pbar:
            mbar = tqdm(
                total=0,
                position=1,
                bar_format="{desc}:{postfix}",
                disable=not self.enable_display_metrics,
            )

            for step in pbar:
                loss = self.training_step(model, step)

                if step % self.progress_bar_interval == 0:
                    pbar.set_postfix({"loss": f"{loss.float():.5f}"})

                optimizer.zero_grad()
                loss.backward()

                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.clip_grad_norm
                    )

                optimizer.step()
                scheduler.step()

                if step % self.log_interval == 0:
                    model.eval()

                    display_metrics = self.logging_step(model, step)

                    mbar.set_description_str(f"Metrics (step {step})")
                    mbar.set_postfix(display_metrics)

                    model.train()

        return
