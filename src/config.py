"""Project configuration: loads config.yaml and resolves paths against the repo root."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT / "config.yaml"


def load_config(path: Path = CONFIG_FILE) -> dict:
    """Read the YAML config into a plain dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str | Path) -> Path:
    """Turn a path from config.yaml into an absolute path under the repo root."""
    return ROOT / relative
