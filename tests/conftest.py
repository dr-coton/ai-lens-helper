"""Pytest configuration ensuring the package under src/ is importable."""

from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:  # pragma: no cover - pytest hook
    project_root = Path(__file__).resolve().parent.parent
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
