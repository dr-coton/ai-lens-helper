"""Backbone registry placeholder."""

from __future__ import annotations

from typing import Callable, Dict

BackboneFactory = Callable[[], object]


def available_backbones() -> Dict[str, BackboneFactory]:
    """Return the registry of supported backbones (stub)."""

    return {
        "vit_b16": lambda: None,
        "convnext_tiny": lambda: None,
        "resnet50": lambda: None,
        "efficientnet_b3": lambda: None,
    }
