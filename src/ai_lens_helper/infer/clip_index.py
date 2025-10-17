"""FAISS-based CLIP embedding index builder and searcher."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from ..models.yolo_clip import YOLOCLIPPipeline


@dataclass
class IndexItem:
    """Single indexed item with embeddings."""

    item_name: str
    place: str
    embeddings: np.ndarray  # Shape: (num_images, embedding_dim)
    mean_embedding: np.ndarray  # Shape: (embedding_dim,)
    image_count: int


class CLIPIndexBuilder:
    """Build FAISS index from dataset using YOLO+CLIP."""

    def __init__(
        self,
        pipeline: YOLOCLIPPipeline,
        embedding_dim: int = 512
    ):
        """
        Initialize index builder.

        Args:
            pipeline: YOLO+CLIP processing pipeline
            embedding_dim: CLIP embedding dimension (512 for ViT-B-16)
        """
        self.pipeline = pipeline
        self.embedding_dim = embedding_dim

    def build_from_directory(
        self,
        data_root: Path,
        place: str,
        image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    ) -> Dict[str, IndexItem]:
        """
        Build index from directory structure: data_root/{place}/{item}/*.jpg

        Args:
            data_root: Root data directory
            place: Place/location identifier
            image_extensions: Supported image file extensions

        Returns:
            Dictionary mapping item_name -> IndexItem
        """
        place_dir = data_root / place
        if not place_dir.exists():
            raise ValueError(f"Place directory not found: {place_dir}")

        index_items: Dict[str, IndexItem] = {}

        for item_dir in sorted(place_dir.iterdir()):
            if not item_dir.is_dir():
                continue

            item_name = item_dir.name
            image_files = [
                f for f in item_dir.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]

            if not image_files:
                print(f"⚠️  Skipping {item_name}: no images found")
                continue

            # Extract embeddings for all images
            embeddings_list = []
            for img_path in image_files:
                try:
                    embedding = self.pipeline.process_image(img_path)
                    embeddings_list.append(embedding)
                except Exception as e:
                    print(f"⚠️  Failed to process {img_path}: {e}")
                    continue

            if not embeddings_list:
                print(f"⚠️  Skipping {item_name}: no valid embeddings")
                continue

            embeddings = np.stack(embeddings_list, axis=0)  # (N, 512)
            mean_embedding = np.mean(embeddings, axis=0)  # (512,)
            mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)  # L2 normalize

            index_items[item_name] = IndexItem(
                item_name=item_name,
                place=place,
                embeddings=embeddings,
                mean_embedding=mean_embedding,
                image_count=len(embeddings_list)
            )

            print(f"✓ Indexed {item_name}: {len(embeddings_list)} images")

        return index_items

    def save_index(
        self,
        index_items: Dict[str, IndexItem],
        output_path: Path,
        place: str,
        metadata: Optional[Dict[str, object]] = None
    ) -> None:
        """
        Save index to disk in compatible format.

        Args:
            index_items: Dictionary of indexed items
            output_path: Output file path (.npz for FAISS, .json for metadata)
            place: Place identifier
            metadata: Optional metadata to include
        """
        # Prepare data structures
        item_names = []
        mean_embeddings = []

        for item_name, item in sorted(index_items.items()):
            item_names.append(item_name)
            mean_embeddings.append(item.mean_embedding)

        mean_embeddings_array = np.stack(mean_embeddings, axis=0)  # (num_items, 512)

        # Build FAISS index
        index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity for normalized vectors)
        index.add(mean_embeddings_array)

        # Save FAISS index
        faiss_path = output_path.with_suffix(".faiss")
        faiss.write_index(index, str(faiss_path))

        # Save metadata
        meta = metadata or {}
        meta.update({
            "place": place,
            "num_items": len(item_names),
            "embedding_dim": self.embedding_dim,
            "index_type": "faiss_flat_ip",
            "model_type": "yolo_clip"
        })

        json_data = {
            "metadata": meta,
            "place": place,
            "items": {
                item_name: {
                    "image_count": index_items[item_name].image_count,
                    "embedding": index_items[item_name].mean_embedding.tolist()
                }
                for item_name in item_names
            },
            "item_names": item_names  # Ordering for FAISS index
        }

        json_path = output_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Index saved:")
        print(f"  FAISS: {faiss_path}")
        print(f"  Metadata: {json_path}")


class CLIPIndexSearcher:
    """Search FAISS index for nearest neighbors."""

    def __init__(self, index_path: Path):
        """
        Load FAISS index and metadata.

        Args:
            index_path: Path to .json metadata file (FAISS index inferred)
        """
        json_path = index_path.with_suffix(".json")
        faiss_path = index_path.with_suffix(".faiss")

        if not json_path.exists():
            raise FileNotFoundError(f"Index metadata not found: {json_path}")
        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {faiss_path}")

        # Load metadata
        with open(json_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.place = self.metadata["place"]
        self.item_names = self.metadata["item_names"]
        self.items = self.metadata["items"]

        # Load FAISS index
        self.index = faiss.read_index(str(faiss_path))

    def search(
        self,
        query_embedding: np.ndarray,
        topk: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Search for nearest neighbors.

        Args:
            query_embedding: Query embedding vector (512-dim, L2 normalized)
            topk: Number of results to return

        Returns:
            List of (item_name, similarity_score) sorted by score descending
        """
        # Ensure L2 normalized
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

        # Search
        scores, indices = self.index.search(query_embedding, topk)

        # Map to item names
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.item_names):
                item_name = self.item_names[idx]
                results.append((item_name, float(score)))

        return results
