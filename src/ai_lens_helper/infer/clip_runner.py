"""Inference runner using YOLO + CLIP + FAISS."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .runner import InferenceResult, Prediction
from .clip_index import CLIPIndexSearcher
from ..models.yolo_clip import YOLOCLIPPipeline


class CLIPInferenceRunner:
    """Perform YOLO+CLIP based inference against FAISS index."""

    def __init__(
        self,
        *,
        model_path: Path,
        device: str = "cpu",
        config: Optional[Dict[str, object]] = None,
        yolo_model: str = "yolov8n.pt",
        clip_model: str = "ViT-B-16",
        clip_pretrained: str = "openai"
    ) -> None:
        """
        Initialize CLIP-based inference runner.

        Args:
            model_path: Path to index .json file
            device: Device to run on ('cpu' or 'cuda')
            config: Optional configuration dict
            yolo_model: YOLO model variant
            clip_model: CLIP model architecture
            clip_pretrained: CLIP pretrained weights
        """
        self.model_path = model_path
        self.device = device
        self.config = config or {}

        # Load index
        self.searcher = CLIPIndexSearcher(model_path)
        self.place = self.searcher.place

        # Initialize YOLO+CLIP pipeline
        self.pipeline = YOLOCLIPPipeline(
            yolo_model=yolo_model,
            clip_model=clip_model,
            clip_pretrained=clip_pretrained,
            device=device
        )

        # Resolve default threshold
        self._default_threshold = self._resolve_default_threshold()

    def _resolve_default_threshold(self) -> float:
        """Resolve reject threshold from config or metadata."""
        # Priority: config > metadata > default
        config_threshold: Optional[float] = None
        if "reject_threshold" in self.config:
            config_threshold = float(self.config["reject_threshold"])  # type: ignore[arg-type]
        elif "reject" in self.config and isinstance(self.config["reject"], dict):
            reject_section = self.config["reject"]
            if "threshold" in reject_section:
                config_threshold = float(reject_section["threshold"])  # type: ignore[arg-type]

        if config_threshold is not None:
            return config_threshold

        metadata_threshold = self.searcher.metadata.get("metadata", {}).get("reject_threshold")
        if metadata_threshold is not None:
            return float(metadata_threshold)

        # Default threshold for cosine similarity (normalized)
        return 0.7

    def _hints(self) -> List[str]:
        """Generate improvement hints for recollect decision."""
        return [
            "Ensure the exhibit fills the frame.",
            "Reduce reflections and glare.",
            "Capture with a steady hand for sharpness.",
            "Try different angles or lighting conditions."
        ]

    def run(
        self,
        *,
        place: str,
        image_path: Path,
        reject_threshold: Optional[float],
        topk: int,
    ) -> InferenceResult:
        """
        Run inference on a single image.

        Args:
            place: Place identifier (must match index)
            image_path: Path to query image
            reject_threshold: Confidence threshold (None = use default)
            topk: Number of top predictions to return

        Returns:
            InferenceResult with predictions and decision
        """
        if place != self.place:
            raise ValueError(
                f"Place mismatch: index is for '{self.place}', but got '{place}'. "
                "Make sure to use the correct index file for the place."
            )

        # Step 1: Extract CLIP embedding (with YOLO cropping)
        query_embedding = self.pipeline.process_image(image_path)

        # Step 2: Search FAISS index
        search_results = self.searcher.search(query_embedding, topk=topk)

        # Step 3: Convert to predictions
        predictions = [
            Prediction(item=item_name, score=score)
            for item_name, score in search_results
        ]

        # Step 4: Make decision
        threshold = reject_threshold if reject_threshold is not None else self._default_threshold
        top_prediction = predictions[0] if predictions else None

        if top_prediction and top_prediction.score >= threshold:
            decision = "accept"
            message = f"Prediction accepted (CLIP similarity: {top_prediction.score:.3f})"
            hints: Optional[List[str]] = None
            selected_item = top_prediction.item
            confidence = top_prediction.score
        else:
            decision = "recollect"
            message = (
                f"Confidence below threshold ({threshold:.2f}); "
                "the image may not match any known exhibit clearly."
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
