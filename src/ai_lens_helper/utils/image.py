"""Utility helpers for working with images without pulling heavy dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

try:  # pragma: no cover - optional dependency
    from PIL import Image  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - executed when Pillow is missing
    Image = None  # type: ignore


def _load_ppm(path: Path) -> Tuple[int, int, List[Tuple[float, float, float]]]:
    """Load an ASCII PPM (P3) image returning width, height, and RGB pixels."""

    tokens: list[str] = []
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens.extend(line.split())

    if not tokens or tokens[0] != "P3":
        raise ValueError("Only ASCII PPM (P3) images are supported without Pillow installed.")

    if len(tokens) < 4:
        raise ValueError("PPM header is incomplete.")

    width = int(tokens[1])
    height = int(tokens[2])
    max_value = int(tokens[3])
    expected_values = width * height * 3
    raw_values = list(map(int, tokens[4:4 + expected_values]))
    if len(raw_values) != expected_values:
        raise ValueError("PPM image does not contain the expected number of pixel values.")

    scale = float(max_value or 1)
    iterator = iter(raw_values)
    pixels = [(next(iterator) / scale, next(iterator) / scale, next(iterator) / scale) for _ in range(width * height)]
    return width, height, pixels


def load_image_rgb(path: Path) -> Tuple[int, int, List[Tuple[float, float, float]]]:
    """Return width, height, and RGB pixels scaled to the [0, 1] range."""

    if Image is not None:
        with Image.open(path) as img:  # type: ignore[attr-defined]
            rgb_image = img.convert("RGB")
            width, height = rgb_image.size
            pixels = [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in rgb_image.getdata()]
            return width, height, pixels

    if path.suffix.lower() == ".ppm":
        return _load_ppm(path)

    raise ModuleNotFoundError(
        "Pillow is required to load this image format. Install the project with the 'dev' extras "
        "or provide ASCII PPM images when running in minimal environments."
    )


def write_solid_ppm(path: Path, color: Tuple[int, int, int], size: Tuple[int, int] = (16, 16)) -> None:
    """Create a simple ASCII PPM image filled with a single colour."""

    width, height = size
    r, g, b = color
    max_value = 255
    lines = ["P3", f"{width} {height}", str(max_value)]
    pixel = f"{r} {g} {b}"
    lines.extend(pixel for _ in range(width * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def mean_color(pixels: Iterable[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """Compute the arithmetic mean colour for a sequence of RGB pixels."""

    total_r = total_g = total_b = 0.0
    count = 0
    for r, g, b in pixels:
        total_r += r
        total_g += g
        total_b += b
        count += 1
    if count == 0:
        raise ValueError("Cannot compute mean colour of an empty pixel sequence.")
    return total_r / count, total_g / count, total_b / count
