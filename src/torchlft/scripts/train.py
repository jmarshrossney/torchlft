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
from torchlft.nflow.logging import Logger
from torchlft.nflow.io import create_training_directory, load_model_from_checkpoint

parser = ArgumentParser(prog="train")

parser.add_argument("model", type=Model)
parser.add_argument("train", type=Trainer, default=lazy_instance(ReverseKLTrainer))
#parser.add_class_arguments(Logger, "log", skip=["train_dir"])

parser.add_argument("--cuda", action=ActionYesNo)
parser.add_argument("--double", action=ActionYesNo)
parser.add_argument("--dry_run", action=ActionYesNo)

parser.add_argument("-o", "--output", type=Path_drw, default=".")
parser.add_argument("-n", "--name", type=str)
parser.add_argument("-c", "--config", action=ActionConfigFile)


def main(config: Namespace) -> None:

    instantiated_config = parser.instantiate_classes(config)
    model = instantiated_config.model
    trainer = instantiated_config.train
    #logger = config.log
    # tester = config.test

    if config.dry_run:
        # NOTE: same as --print-config
        print(parser.dump(config))
        _ = model(1)
        print(model)
        return

    
    train_dir = create_training_directory(
            path=config.output,
            config=config,
            parser=parser,
            name=config.name,
    )
    logger = Logger(train_dir)

    device = "cuda" if config.cuda else "cpu"
    dtype = torch.float64 if config.double else torch.float32

    model = model.to(device=device, dtype=dtype)

    trainer.train(model, logger)

    model = model.to("cpu")

    train_dir.save_checkpoint(model, trainer.n_steps)

    return model, logger, train_dir

