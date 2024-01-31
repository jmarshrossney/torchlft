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

    def list_checkpoints(self) -> list[int]:
        files = list(self.checkpoint_dir.glob("step_*.ckpt"))
        steps = [f.stem.strip("step_") for f in files]
        return list(sorted(steps))

    def last_checkpoint(self) -> int | None:
        checkpoints = self.list_checkpoints()
        return None if not checkpoints else max(checkpoints)

    def save_checkpoint(
        self, model, step: int, overwrite_existing: bool = False
    ) -> None:
        self.checkpoint_dir.mkdir(parents=False, exist_ok=True)
        ckpt_file = self.checkpoint_dir / f"step_{step}.ckpt"

        #assert self.last_checkpoint is None or step > self._last_checkpoint

        if overwrite_existing and self._last_checkpoint is not None:
            pass  # TODO: delete existing checkpoint using shutil probably

        logger.info(f"Saving model weights to {ckpt_file}")
        torch.save(model.state_dict(), ckpt_file)

    def load_checkpoint(self, step: int | None = None):
        step = step or self.last_checkpoint()
        checkpoint_file = self.checkpoint_dir / f"step_{step}.ckpt"
        logger.info(f"Loading checkpoint from {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file)
        return checkpoint


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

    train_dir = TrainingDirectory(root)

    config_str = parser.dump(config, skip_none=False)

    header = "# Run on {ts} using torchlft v{v}, commit {cm}".format(
        ts=timestamp.strftime("%Y-%m-%d %H:%M"),
        v=get_version(),
        cm=get_commit(),
    )
    config_str = header + "\n" + config_str

    config_file = train_dir.config_file
    logger.info(f"Saving config to {config_file}")
    with config_file.open("w") as file:
        file.write(config_str)

    return train_dir


def load_model_from_checkpoint(
    train_dir: TrainingDirectory,
    parser: ArgumentParser,
    step: int | None = None,
) -> Model:
    # TODO: allow modifications to config when loading?
    logger.info(f"Loading training config from {train_dir.config_file}")
    config = parser.parse_path(train_dir.config_file)
    config = parser.instantiate_classes(config)
    model = config.model

    checkpoint = train_dir.load_checkpoint(step)
    model.load_state_dict(checkpoint)

    return model
