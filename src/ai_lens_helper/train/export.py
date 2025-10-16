"""Model export scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ExportOptions:
    """Options describing how the checkpoint should be exported."""

    ckpt_path: Path
    onnx_path: Optional[Path]
    opset: int = 17


class ModelExporter:
    """Thin wrapper used by the CLI to keep the flow testable."""

    def __init__(self, *, options: ExportOptions) -> None:
        self.options = options

    def export(self) -> str:
        """Return a textual status message until the exporter is implemented."""

        message = (
            "Export pipeline is not ready yet. "
            f"Received checkpoint at '{self.options.ckpt_path}' with opset {self.options.opset}."
        )
        if self.options.onnx_path:
            message += f" ONNX output will be written to '{self.options.onnx_path}'."
        return message
