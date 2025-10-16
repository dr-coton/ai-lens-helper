"""Dataset loading and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm"}


@dataclass
class PlaceSummary:
    """Aggregate statistics for a single place folder."""

    place: str
    item_count: int
    items_meeting_requirement: int


@dataclass
class ValidationReport:
    """Container describing dataset validation results."""

    place_summaries: List[PlaceSummary]
    issues: List[str]


class DataValidator:
    """Lightweight dataset validator focusing on the README layout requirements."""

    def __init__(self, *, data_root: Path, min_images: int = 10) -> None:
        self.data_root = data_root
        self.min_images = min_images

    def _iter_places(self) -> List[Path]:
        return [p for p in sorted(self.data_root.iterdir()) if p.is_dir()]

    def _iter_items(self, place_dir: Path) -> List[Path]:
        return [p for p in sorted(place_dir.iterdir()) if p.is_dir()]

    def _count_images(self, item_dir: Path) -> int:
        return sum(1 for path in item_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)

    def validate(self) -> ValidationReport:
        issues: List[str] = []
        place_summaries: List[PlaceSummary] = []

        if not self.data_root.exists():
            issues.append(f"Data root '{self.data_root}' does not exist.")
            return ValidationReport(place_summaries=[], issues=issues)

        places = self._iter_places()
        if not places:
            issues.append("No place directories found. Expecting 'data/<place>/<item>/*.jpg'.")

        for place_dir in places:
            items = self._iter_items(place_dir)
            if not items:
                issues.append(f"Place '{place_dir.name}' contains no exhibit folders.")
                place_summaries.append(PlaceSummary(place=place_dir.name, item_count=0, items_meeting_requirement=0))
                continue

            meets_requirement = 0
            for item_dir in items:
                image_count = self._count_images(item_dir)
                if image_count < self.min_images:
                    issues.append(
                        f"Item '{place_dir.name}/{item_dir.name}' has {image_count} images (< {self.min_images})."
                    )
                else:
                    meets_requirement += 1
            place_summaries.append(
                PlaceSummary(place=place_dir.name, item_count=len(items), items_meeting_requirement=meets_requirement)
            )

        return ValidationReport(place_summaries=place_summaries, issues=issues)
