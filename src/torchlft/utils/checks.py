from os import PathLike
from pathlib import Path


def is_existing_directory(path: str | PathLike) -> bool:
    path = path if isinstance(path, Path) else Path(path)
    path = path.resolve()
    return path.is_dir()
