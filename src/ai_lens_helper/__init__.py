"""Top level package for the ai-lens-helper project."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .infer.runner import InferenceResult, InferenceRunner
from .utils.io import load_yaml_config

__all__ = ["Lens", "InferenceResult", "InferenceRunner"]


class Lens:
    """High level SDK facade used in the README quickstart example.

    The SDK keeps the surface area minimal while the rest of the package grows.
    At this early stage the implementation simply wires together the inference
    runner and performs basic configuration loading so downstream code already
    has the right extension points available.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: Optional[str] = None,
        config_path: Optional[str | Path] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device or "cpu"
        self.config: Dict[str, Any] = {}
        if config_path is not None:
            self.config = load_yaml_config(config_path)
        if overrides:
            self.config.update(overrides)
        self.runner = InferenceRunner(model_path=self.model_path, device=self.device, config=self.config)

    def infer(
        self,
        *,
        place: str,
        image_path: str | Path,
        reject_threshold: Optional[float] = None,
        topk: int = 3,
    ) -> InferenceResult:
        """Run inference for a single image.

        Parameters
        ----------
        place:
            Museum room or area identifier used to narrow the search space.
        image_path:
            Path to the captured image that should be classified.
        reject_threshold:
            Optional override for the reject threshold. When omitted the
            configured default is used.
        topk:
            Number of candidate predictions that should be returned.
        """

        return self.runner.run(
            place=place,
            image_path=Path(image_path),
            reject_threshold=reject_threshold,
            topk=topk,
        )
