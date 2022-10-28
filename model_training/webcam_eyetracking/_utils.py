from itertools import chain
from pathlib import Path


def get_project_root() -> Path:
    cwd = Path.cwd()
    for path in chain((cwd,), cwd.parents):
        if (path / "pyproject.toml").exists():
            return path
    else:
        raise FileNotFoundError("Could not find project root.")
