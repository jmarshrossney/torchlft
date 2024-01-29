from jsonargparse import (
    ActionConfigFile,
    ActionYesNo,
    ArgumentParser,
    Namespace,
)
from jsonargparse.typing import Path_drw
import torch

from torchlft_experiments.io import TrainingDirectory
from torchlft_experiments.a1.model import Model
from torchlft_experiments.a1.trainer import Trainer

parser = ArgumentParser(prog="train")

parser.add_class_arguments(Model, "model")
parser.add_class_arguments(Trainer, "train")
parser.add_argument("--cuda", action=ActionYesNo)
parser.add_argument("--double", action=ActionYesNo)

parser.add_argument("-o", "--output", type=Path_drw, default=".")
parser.add_argument("-c", "--config", action=ActionConfigFile)

# TODO: --dry-run


def main(config: Namespace) -> None:
    print(config)

    TrainingDirectory.parser = parser
    train_dir = TrainingDirectory.new(config.output, config)

    config = parser.instantiate_classes(config)
    model = config.model
    trainer = config.train
    # tester = config.test

    device = "cuda" if config.cuda else "cpu"
    dtype = torch.float64 if config.double else torch.float32

    model = model.to(device=device, dtype=dtype)

    training_log = trainer.train(model)  # add training directory?

    model = model.to("cpu")
