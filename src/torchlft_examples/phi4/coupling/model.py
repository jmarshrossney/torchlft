import pytorch_lightning as pl
import torch

from torchlft.nflow import BoltzmannGenerator
from torchlft.lightning.metrics import LogWeightMetrics

from torchlft.typing import Tensor

class Model(BoltzmannGenerator, pl.LightningModule):

    def __init__(self, base, target, flow):
        super().__init__(base, target, flow)

    def training_step(self, inputs: tuple[Tensor, Tensor]) -> Tensor:
        outputs = self.sampling_step(inputs)
        log_weights = outputs["model_action"] - outputs["target_action"]
        loss = log_weights.negative().mean(dim=0, keepdim=True)
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, inputs: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        outputs = self.sampling_step(inputs)
        log_weights = outputs["model_action"] - outputs["target_action"]
        metrics = LogWeightMetrics(log_weights)
        self.log_dict(metrics.asdict(), on_step=False, on_epoch=True)

    def train_dataloader(self):
        def generator():
            while True:
                x = self.base.sample(1000)
                a = self.base.compute(x)
                yield (x, a)

        return generator()

    def val_dataloader(self):
        return self.train_dataloader()


"""class Model(pl.LightningModule):

    def __init__(
            self,
"""
