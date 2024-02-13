from jsonargparse import (
    ActionConfigFile,
    ActionYesNo,
    ArgumentParser,
    lazy_instance,
    Namespace,
)
from jsonargparse.typing import Path_drw
import torch

from torchlft.nflow.model import Model
from torchlft.nflow.train import Trainer, ReverseKLTrainer
from torchlft.nflow.logging import Logger, DefaultLogger
from torchlft.nflow.io import create_training_directory

parser = ArgumentParser(prog="train")

parser.add_argument("model", type=Model)
parser.add_argument(
    "train", type=Trainer, default=lazy_instance(ReverseKLTrainer)
)
parser.add_subclass_arguments(
    Logger, "log", default=lazy_instance(DefaultLogger)
)

parser.add_argument("--cuda", action=ActionYesNo, default=False)
parser.add_argument("--double", action=ActionYesNo, default=False)
parser.add_argument("--output", action=ActionYesNo, default=True)

parser.add_argument("-o", "--output_path", type=Path_drw, default=".")
parser.add_argument("-n", "--output_name", type=str)
parser.add_argument("-c", "--config", action=ActionConfigFile)

# parser.link_arguments("log.train_dir",


def main(config: Namespace) -> None:
    instantiated_config = parser.instantiate_classes(config)
    model = instantiated_config.model
    trainer = instantiated_config.train
    logger = instantiated_config.log
    # tester = config.test

    if config.output:
        train_dir = create_training_directory(
            path=config.output_path,
            config=config,
            parser=parser,
            name=config.output_name,
        )
        logger.train_dir = train_dir

    device = "cuda" if config.cuda else "cpu"
    dtype = torch.float64 if config.double else torch.float32

    model = model.to(device=device, dtype=dtype)

    trainer.train(model, logger)

    model = model.to("cpu")

    if config.output:
        train_dir.save_checkpoint(model, trainer.n_steps)

    return model, logger
