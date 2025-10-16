"""Loss function registry placeholder."""

from __future__ import annotations

from typing import Callable, Dict

LossFactory = Callable[[], object]


def available_losses() -> Dict[str, LossFactory]:
    """Return the registry of supported loss functions (stub)."""

    return {
        "ce": lambda: None,
        "arcface": lambda: None,
        "triplet": lambda: None,
    }
