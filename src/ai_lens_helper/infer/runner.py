"""Lightweight inference runner based on colour-matching heuristics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..utils.image import load_image_rgb, mean_color
from ..utils.io import load_json


@dataclass
class Prediction:
    """Single prediction entry."""

    item: str
    score: float


@dataclass
class InferenceResult:
    """Structured inference response compatible with the README examples."""

    place: str
    predictions: List[Prediction] = field(default_factory=list)
    decision: str = "recollect"
    selected_item: Optional[str] = None
    confidence: Optional[float] = None
    message: Optional[str] = None
    hints: Optional[List[str]] = None

    def as_dict(self) -> Dict[str, object]:
        """Convert the dataclass to a serialisable dictionary."""

        return {
            "place": self.place,
            "predictions": [prediction.__dict__ for prediction in self.predictions],
            "decision": self.decision,
            "selected_item": self.selected_item,
            "confidence": self.confidence,
            "message": self.message,
            "hints": self.hints,
        }


@dataclass
class InferenceOptions:
    """Options consumed by the CLI single-image entry point."""

    place: str
    image_path: Path
    reject_threshold: Optional[float]
    topk: int


@dataclass
class BatchInferenceOptions:
    """Options consumed by the CLI batch entry point."""

    model_path: Path
    place: str
    input_dir: Path
    output_path: Path
    reject_threshold: Optional[float]
    topk: int

    def summary_message(self) -> str:
        """Produce a human readable summary used by the CLI stub."""

        threshold = self.reject_threshold if self.reject_threshold is not None else "default"
        return (
            "Batch inference is not implemented yet. "
            f"Would process images from '{self.input_dir}' for place '{self.place}' "
            f"into '{self.output_path}' (threshold={threshold}, topk={self.topk})."
        )


MAX_COLOR_DISTANCE = math.sqrt(3.0)


class InferenceRunner:
    """Perform colour mean matching against a pre-built index."""

    def __init__(self, *, model_path: Path, device: str, config: Optional[Dict[str, object]] = None) -> None:
        self.model_path = model_path
        self.device = device
        self.config = config or {}
        self._places, self._metadata = self._load_index(model_path)
        self._default_threshold = self._resolve_default_threshold()

    def _resolve_default_threshold(self) -> float:
        config_threshold: Optional[float] = None
        if "reject_threshold" in self.config:
            config_threshold = float(self.config["reject_threshold"])  # type: ignore[arg-type]
        elif "reject" in self.config and isinstance(self.config["reject"], dict):
            reject_section = self.config["reject"]
            if "threshold" in reject_section:
                config_threshold = float(reject_section["threshold"])  # type: ignore[arg-type]

        if config_threshold is not None:
            return config_threshold

        metadata_threshold = self._metadata.get("reject_threshold")
        if metadata_threshold is not None:
            return float(metadata_threshold)

        return 0.62

    def _load_index(
        self, model_path: Path
    ) -> Tuple[Dict[str, Dict[str, Tuple[Tuple[float, float, float], int]]], Dict[str, object]]:
        data = load_json(model_path)
        metadata = data.get("metadata", {})

        if "places" in data:
            place_payload = data["places"]
        elif "place" in data and "items" in data:
            place_payload = {data["place"]: {"items": data["items"]}}
        else:
            raise ValueError("Index file must contain either 'places' or ('place' and 'items') entries.")

        parsed: Dict[str, Dict[str, Tuple[Tuple[float, float, float], int]]] = {}
        for place, payload in place_payload.items():
            items_section = payload.get("items", payload)
            if not isinstance(items_section, dict):
                raise ValueError(f"Items for place '{place}' must be a mapping.")
            parsed_items: Dict[str, Tuple[Tuple[float, float, float], int]] = {}
            for item_name, item_payload in items_section.items():
                if "mean_color" in item_payload:
                    raw_vector = item_payload["mean_color"]
                elif "feature" in item_payload:
                    raw_vector = item_payload["feature"]
                else:
                    raise ValueError(
                        f"Item '{item_name}' in place '{place}' does not contain a recognised feature key."
                    )
                vector = [float(value) for value in raw_vector]
                if any(value > 1.5 for value in vector):
                    vector = [value / 255.0 for value in vector]
                if len(vector) != 3:
                    raise ValueError(
                        f"Item '{item_name}' in place '{place}' must provide exactly three colour components."
                    )
                image_count = int(item_payload.get("image_count", 0))
                parsed_items[item_name] = ((vector[0], vector[1], vector[2]), image_count)
            if not parsed_items:
                raise ValueError(f"No usable items found for place '{place}'.")
            parsed[place] = parsed_items

        return parsed, metadata

    def _mean_rgb(self, image_path: Path) -> Tuple[float, float, float]:
        _width, _height, pixels = load_image_rgb(image_path)
        return mean_color(pixels)

    def _score(
        self, reference: Tuple[float, float, float], sample: Tuple[float, float, float]
    ) -> float:
        distance = math.sqrt(sum((ref - samp) ** 2 for ref, samp in zip(reference, sample)))
        normalised = max(0.0, 1.0 - min(distance, MAX_COLOR_DISTANCE) / MAX_COLOR_DISTANCE)
        return round(normalised, 4)

    def _hints(self) -> List[str]:
        return [
            "Ensure the exhibit fills the frame.",
            "Reduce reflections and glare.",
            "Capture with a steady hand for sharpness.",
        ]

    def run(
        self,
        *,
        place: str,
        image_path: Path,
        reject_threshold: Optional[float],
        topk: int,
    ) -> InferenceResult:
        if place not in self._places:
            known_places = ", ".join(sorted(self._places))
            raise ValueError(f"Place '{place}' is not present in the index. Known places: {known_places}")

        sample_vector = self._mean_rgb(image_path)
        predictions: List[Prediction] = []
        for item, (reference_vector, _count) in self._places[place].items():
            predictions.append(Prediction(item=item, score=self._score(reference_vector, sample_vector)))

        predictions.sort(key=lambda pred: pred.score, reverse=True)
        predictions = predictions[:topk]

        threshold = reject_threshold if reject_threshold is not None else self._default_threshold
        top_prediction = predictions[0] if predictions else None

        if top_prediction and top_prediction.score >= threshold:
            decision = "accept"
            message = "Prediction accepted based on colour similarity heuristic."
            hints: Optional[List[str]] = None
            selected_item = top_prediction.item
            confidence = top_prediction.score
        else:
            decision = "recollect"
            message = (
                "Confidence below threshold; capture conditions may differ from the reference dataset."
            )
            hints = self._hints()[:topk]
            selected_item = top_prediction.item if top_prediction else None
            confidence = top_prediction.score if top_prediction else None

        return InferenceResult(
            place=place,
            predictions=predictions,
            decision=decision,
            selected_item=selected_item,
            confidence=confidence,
            message=message,
            hints=hints,
        )
