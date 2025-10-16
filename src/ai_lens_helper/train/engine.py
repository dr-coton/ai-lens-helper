"""Training engine scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..utils.io import load_yaml_config


@dataclass
class TrainingOptions:
    """Configuration holder passed from the CLI to the engine."""

    config_path: Optional[Path]
    data_root: Path
    output_dir: Path


class TrainingEngine:
    """Placeholder training engine that will grow with future tasks."""

    def __init__(self, *, options: TrainingOptions) -> None:
        self.options = options
        self.config = {}
        if options.config_path is not None:
            self.config = load_yaml_config(options.config_path)

    def train(self) -> str:
        """Return a helpful placeholder message for now."""

        self.output_dir = self.options.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return (
            "Training pipeline is not implemented yet. "
            "Scaffolding created the output directory and loaded configuration successfully."
        )
