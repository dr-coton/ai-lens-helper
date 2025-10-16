"""Evaluation utilities placeholder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class EvaluationMetric:
    """Represents a single evaluation metric result."""

    name: str
    value: float
    description: str


def summarize_metrics(metrics: List[EvaluationMetric]) -> Dict[str, float]:
    """Convert a list of metric objects into a serialisable dictionary."""

    return {metric.name: metric.value for metric in metrics}
