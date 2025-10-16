"""Image quality assessment stubs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class QualityFlag:
    """Represents a potential issue detected in the input image."""

    name: str
    description: str


def default_quality_checks() -> List[QualityFlag]:
    """Return the default quality hints mentioned in the README."""

    return [
        QualityFlag(name="framing", description="Ensure the exhibit is centred in the frame."),
        QualityFlag(name="glare", description="Minimise reflections and glare when shooting."),
        QualityFlag(name="sharpness", description="Avoid motion blur by keeping the camera steady."),
    ]
