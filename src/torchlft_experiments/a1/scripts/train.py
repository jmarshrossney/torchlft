from jsonargparse import (
    ActionConfigFile,
    ActionYesNo,
    ArgumentParser,
    Namespace,
)
from jsonargparse.typing import Path_drw
import torch

from torchlft_experiments.a1.model import Model
from torchlft_experiments.a1.trainer import Trainer, Logger
from torchlft.nflow.io import create_training_directory, load_model_from_checkpoint

parser = ArgumentParser(prog="train")

parser.add_class_arguments(Model, "model")
parser.add_class_arguments(Trainer, "train")
#parser.add_class_arguments(Logger, "log", skip=["train_dir"])
parser.add_argument("--cuda", action=ActionYesNo)
parser.add_argument("--double", action=ActionYesNo)

parser.add_argument("-o", "--output", type=Path_drw, default=".")
parser.add_argument("-c", "--config", action=ActionConfigFile)

# TODO: --dry-run


def main(config: Namespace) -> None:
    print(config)
    
    train_dir = create_training_directory(
            path=config.output,
            config=config,
            parser=parser,
    )

    config = parser.instantiate_classes(config)
    model = config.model
    trainer = config.train
    #logger = config.log
    # tester = config.test
    logger = Logger(train_dir)

    device = "cuda" if config.cuda else "cpu"
    dtype = torch.float64 if config.double else torch.float32

    model = model.to(device=device, dtype=dtype)

    trainer.train(model, logger)

    model = model.to("cpu")

    d = logger.as_dict()

    train_dir.save_checkpoint(model, trainer.n_steps)

    #print({k: v.mean(dim=1) for k, v in d.items()})

