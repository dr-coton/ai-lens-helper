"""Classification and embedding head stubs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class HeadConfig:
    """Generic configuration shared by future head implementations."""

    name: str
    embedding_dim: int
    num_classes: int


def create_softmax_head(config: HeadConfig) -> Callable[[], object]:
    """Return a factory compatible with the training engine."""

    def factory() -> object:
        raise NotImplementedError("Softmax head is not implemented yet.")

    return factory
