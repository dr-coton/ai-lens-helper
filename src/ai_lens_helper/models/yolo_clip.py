"""YOLO + CLIP based exhibit detector and embedder."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
import open_clip


class YOLODetector:
    """Wrapper for YOLO object detection."""

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.25, device: str = "cpu"):
        """
        Initialize YOLO detector.

        Args:
            model_name: YOLO model variant (yolov8n/s/m/l/x)
            confidence: Detection confidence threshold
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.device = device

    def detect_objects(self, image_path: Path) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect objects in image.

        Args:
            image_path: Path to input image

        Returns:
            List of (x1, y1, x2, y2, confidence) bounding boxes
        """
        results = self.model.predict(
            source=str(image_path),
            conf=self.confidence,
            device=self.device,
            verbose=False
        )

        boxes = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = float(box.conf[0].cpu().numpy())
                    boxes.append((
                        int(xyxy[0]), int(xyxy[1]),
                        int(xyxy[2]), int(xyxy[3]),
                        conf
                    ))

        return boxes

    def crop_largest_object(self, image_path: Path) -> Optional[Image.Image]:
        """
        Detect and crop the largest object from image.

        Args:
            image_path: Path to input image

        Returns:
            Cropped PIL Image or None if no detection
        """
        boxes = self.detect_objects(image_path)

        if not boxes:
            # No detection - return full image
            return Image.open(image_path).convert("RGB")

        # Find largest box by area
        largest_box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        x1, y1, x2, y2, _ = largest_box

        # Crop image
        image = Image.open(image_path).convert("RGB")
        cropped = image.crop((x1, y1, x2, y2))

        return cropped


class CLIPEmbedder:
    """Wrapper for CLIP image embedding extraction."""

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        device: str = "cpu"
    ):
        """
        Initialize CLIP embedder.

        Args:
            model_name: CLIP model architecture
            pretrained: Pretrained weights source
            device: Device to run on ('cpu' or 'cuda')
        """
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device
        )
        self.model.eval()

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """
        Extract CLIP embedding from PIL Image.

        Args:
            image: PIL Image (RGB)

        Returns:
            Normalized embedding vector (512-dim for ViT-B-16)
        """
        with torch.no_grad():
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            features = self.model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)  # L2 normalize
            return features.cpu().numpy()[0]

    def embed_image_path(self, image_path: Path) -> np.ndarray:
        """
        Extract CLIP embedding from image path.

        Args:
            image_path: Path to image file

        Returns:
            Normalized embedding vector
        """
        image = Image.open(image_path).convert("RGB")
        return self.embed_image(image)


class YOLOCLIPPipeline:
    """Combined YOLO detection + CLIP embedding pipeline."""

    def __init__(
        self,
        yolo_model: str = "yolov8n.pt",
        clip_model: str = "ViT-B-16",
        clip_pretrained: str = "openai",
        device: str = "cpu",
        yolo_confidence: float = 0.25
    ):
        """
        Initialize YOLO+CLIP pipeline.

        Args:
            yolo_model: YOLO model variant
            clip_model: CLIP model architecture
            clip_pretrained: CLIP pretrained weights
            device: Device to run on
            yolo_confidence: YOLO detection confidence threshold
        """
        self.device = device
        self.detector = YOLODetector(
            model_name=yolo_model,
            confidence=yolo_confidence,
            device=device
        )
        self.embedder = CLIPEmbedder(
            model_name=clip_model,
            pretrained=clip_pretrained,
            device=device
        )

    def process_image(self, image_path: Path) -> np.ndarray:
        """
        Detect largest object and extract CLIP embedding.

        Args:
            image_path: Path to input image

        Returns:
            CLIP embedding vector (L2 normalized)
        """
        # Step 1: YOLO detection and crop
        cropped_image = self.detector.crop_largest_object(image_path)

        if cropped_image is None:
            # Fallback: use full image
            cropped_image = Image.open(image_path).convert("RGB")

        # Step 2: CLIP embedding
        embedding = self.embedder.embed_image(cropped_image)

        return embedding
