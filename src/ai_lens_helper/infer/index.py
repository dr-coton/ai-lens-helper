"""Utility for constructing a lightweight colour-based retrieval index."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from ..train.datamodule import IMAGE_EXTENSIONS
from ..utils.image import load_image_rgb, mean_color
from ..utils.io import dump_json


@dataclass
class IndexOptions:
    """Parameters required to build a retrieval index."""

    model_path: Path
    data_root: Path
    place: str
    save_path: Path


@dataclass
class IndexBuildSummary:
    """Short summary describing the generated index."""

    place: str
    save_path: Path
    item_count: int
    image_count: int


class IndexBuilder:
    """Create a trivial colour-mean retrieval index for prototyping."""

    def __init__(self, *, options: IndexOptions) -> None:
        self.options = options

    def _iter_item_dirs(self, place_dir: Path) -> Iterable[Path]:
        for item_dir in sorted(place_dir.iterdir()):
            if item_dir.is_dir():
                yield item_dir

    def _iter_images(self, item_dir: Path) -> List[Path]:
        images: List[Path] = []
        for image_path in sorted(item_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS and image_path.is_file():
                images.append(image_path)
        return images

    def _mean_rgb(self, image_paths: Iterable[Path]) -> Tuple[float, float, float]:
        means: List[Tuple[float, float, float]] = []
        for image_path in image_paths:
            _width, _height, pixels = load_image_rgb(image_path)
            means.append(mean_color(pixels))
        if not means:
            raise ValueError("At least one valid image is required to compute embeddings.")
        count = len(means)
        sum_r = sum(value[0] for value in means)
        sum_g = sum(value[1] for value in means)
        sum_b = sum(value[2] for value in means)
        return sum_r / count, sum_g / count, sum_b / count

    def build(self) -> IndexBuildSummary:
        """Compute mean RGB descriptors for every item within the requested place."""

        place_dir = self.options.data_root / self.options.place
        if not place_dir.exists():
            raise FileNotFoundError(
                f"Place '{self.options.place}' does not exist under '{self.options.data_root}'."
            )

        items_payload: Dict[str, Dict[str, object]] = {}
        total_images = 0

        for item_dir in self._iter_item_dirs(place_dir):
            image_paths = self._iter_images(item_dir)
            if not image_paths:
                continue
            descriptor = self._mean_rgb(image_paths)
            total_images += len(image_paths)
            items_payload[item_dir.name] = {
                "mean_color": [descriptor[0], descriptor[1], descriptor[2]],
                "image_count": len(image_paths),
            }

        if not items_payload:
            raise ValueError(
                "No items with supported images were found. Ensure the dataset matches the expected layout."
            )

        payload = {
            "version": 1,
            "place": self.options.place,
            "representation": "mean_rgb",
            "metadata": {"reject_threshold": 0.62},
            "items": items_payload,
        }
        dump_json(payload, self.options.save_path)

        return IndexBuildSummary(
            place=self.options.place,
            save_path=self.options.save_path,
            item_count=len(items_payload),
            image_count=total_images,
        )
