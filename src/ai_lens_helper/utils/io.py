"""I/O helpers used across the package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file if it exists."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file '{config_path}' does not exist.")
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional dependency
        raise ModuleNotFoundError(
            "PyYAML is required to load configuration files. Install the project with the 'dev' "
            "extras or add PyYAML to your environment."
        ) from exc
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a mapping at the top level.")
    return data


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load a JSON file returning a dictionary."""

    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file '{json_path}' does not exist.")
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("JSON file must contain an object at the top level.")
    return data


def dump_json(data: Dict[str, Any], path: str | Path) -> None:
    """Write a JSON file with UTF-8 encoding."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
