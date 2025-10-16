"""Visualization helpers placeholder."""

from __future__ import annotations

from pathlib import Path


def save_confusion_matrix(path: Path) -> None:
    """Placeholder implementation for future reporting utilities."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Confusion matrix plotting is not implemented yet.\n", encoding="utf-8")
