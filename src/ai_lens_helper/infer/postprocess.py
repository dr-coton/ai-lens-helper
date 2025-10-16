"""Post-processing helpers placeholder."""

from __future__ import annotations

from typing import Iterable, List

from .runner import Prediction


def topk_predictions(predictions: Iterable[Prediction], k: int) -> List[Prediction]:
    """Return the top-k predictions sorted by score."""

    return sorted(predictions, key=lambda pred: pred.score, reverse=True)[:k]
