from datetime import datetime
import importlib.metadata
from os import PathLike
from pathlib import Path
import logging
import subprocess

import torch
from jsonargparse import ArgumentParser, Namespace

from torchlft.nflow.model import Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingDirectory:
    def __init__(self, root: str | Path):
        root = root if isinstance(root, Path) else Path(root)
        root = root.resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"No directory found at {root}")

        self._root = root
        self._last_checkpoint = None

    @property
    def root(self) -> Path:
        return self._root

    @property
    def checkpoint_dir(self) -> Path:
        return self._root / "checkpoints"

    @property
    def log_dir(self) -> Path:
        return self._root / "training_logs"

    @property
    def config_file(self) -> Path:
        return self._root / "config.yaml"

    def save_checkpoint(
        self, model, step: int, overwrite_existing: bool = False
    ) -> None:
        ckpt_file = self.checkpoint_dir / f"step_{step}.ckpt"

        assert self._last_checkpoint is None or step > self._last_checkpoint

        if overwrite_existing and self._last_checkpoint is not None:
            pass  # TODO: delete existing checkpoint using shutil probably

        logger.info(f"Saving model weights to {ckpt_file}")
        torch.save(model.state_dict(), ckpt_file)

        self._last_checkpoint = step

    def load_checkpoint(self, step: int | None = None):
        # TODO: allow modifications to config when loading?
        # TODO: load parser

        step = step if step is not None else self._last_checkpoint

        logger.info(f"Loading training config from {self.config_file}")
        config = parser.parse_path(self.config_file)
        config = parser.instantiate_classes(config)
        model = config.model

        checkpoint_file = self.checkpoint_dir / f"step_{step}.ckpt"
        logger.info(f"Loading checkpoint from {checkpoint_file}")
        checkpoint = torch.load(self.checkpoint_file)
        model.load_state_dict(checkpoint)

        return model


def get_version():
    return importlib.metadata.version("torchlft")


def get_commit():
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parent),
                "rev-parse",
                "--short",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.info(f"Unable to obtain git commit hash: {e}")
        return "unknown"
    else:
        return result.stdout.strip()


def create_training_directory(
    path: str | PathLike,
    config: Namespace,
    parser: ArgumentParser,
) -> TrainingDirectory:
    path = path if isinstance(path, Path) else Path(str(path))
    path = path.resolve()

    timestamp = datetime.now()

    root = path / "train_{ts}".format(ts=timestamp.strftime("%Y%m%d%H%M%S"))
    logger.info(f"Creating new directory at {root}")
    root.mkdir(exist_ok=False, parents=True)

    config_str = parser.dump(config, skip_none=False)

    header = "# Run on {ts} using torchlft v{v}, commit {cm}".format(
        ts=timestamp.strftime("%Y-%m-%d %H:%M"),
        v=get_version(),
        cm=get_commit(),
    )
    config_str = header + "\n" + config_str

    config_file = root / "config.yaml"
    logger.info(f"Saving config to {config_file}")
    with config_file.open("w") as file:
        file.write(config_str)

    return TrainingDirectory(root)


def load_model_from_checkpoint(
    train_dir: TrainingDirectory,
    parser: ArgumentParser,
    step: int | None = None,
) -> Model:
    logger.info(f"Loading training config from {train_dir.config_file}")
    config = parser.parse_path(train_dir.config_file)
    config = parser.instantiate_classes(config)
    model = config.model

    checkpoint = train_dir.load_checkpoint(step)
    model.load_state_dict(checkpoint)

    return model
