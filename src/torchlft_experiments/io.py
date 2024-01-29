from datetime import datetime
import importlib.metadata
from os import PathLike
from pathlib import Path
import logging
import subprocess
from typing import ClassVar, Self

import torch
from jsonargparse import Namespace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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



class _ExistingDirectory:
    def __init__(self, root: str | Path):
        root = root if isinstance(root, Path) else Path(root)
        root = root.resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"No directory found at {root}")

        self._root = root

    @property
    def root(self) -> Path:
        return self._root


class TrainingDirectory(_ExistingDirectory):
    parser: ClassVar  # todo

    @classmethod
    def new(
        cls,
        path: str | PathLike,
        config: Namespace,
    ) -> Self:
        path = path if isinstance(path, Path) else Path(str(path))
        path = path.resolve()

        timestamp = datetime.now()

        dirname = "train_{ts}".format(ts=timestamp.strftime("%Y%m%d%H%M%S"))

        root = path / dirname
        logger.info(f"Creating new directory at {root}")
        root.mkdir(exist_ok=False, parents=True)

        # TODO Get parser
        parser = cls.parser

        print(config)
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

        return cls(root)

    def save_checkpoint(
        self, model, step: int, overwrite_existing: bool = False
    ) -> None:
        ckpt_file = self.root / "checkpoints" / f"step_{step}.ckpt"

        assert step > self._last_checkpoint

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

        checkpoint_file = self.root / "checkpoints" / f"step_{step}.ckpt"
        logger.info(f"Loading checkpoint from {checkpoint_file}")
        checkpoint = torch.load(self.checkpoint_file)
        model.load_state_dict(checkpoint)

        return model

    def log(self, step: int) -> None:
        raise NotImplementedError
