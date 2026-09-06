"""
Base adapter interface (Section 3.4): canonical Tile -> model input,
model output -> canonical Evidence. Every model's quirks live inside its
own adapter and nowhere else — registry.py, loader.py, and inference.py
never branch on which specific model is running.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from backend.shared.schemas import Evidence, Modality, ModelRegistryEntry, Tile


class BaseAdapter(ABC):
    """One adapter instance per call; construct with per-call context (query, etc.)."""

    #: TaskTypes this adapter knows how to handle — a sanity guard against
    #: inference.py's task->adapter map drifting from what the adapter
    #: actually implements.
    handles: tuple = ()

    @abstractmethod
    def preprocess(self, tiles: list[Tile], entry: ModelRegistryEntry) -> Any:
        """Canonical Tile list -> whatever input shape the model needs."""

    @abstractmethod
    def infer(self, model: Any, model_input: Any) -> Any:
        """The actual forward pass. Nothing here touches Evidence or Tile."""

    @abstractmethod
    def postprocess(
        self,
        raw_output: Any,
        tiles: list[Tile],
        entry: ModelRegistryEntry,
        modality_used: list[Modality],
    ) -> Evidence:
        """
        Model output -> canonical Evidence. Coordinates MUST be converted to
        full-image pixel space here (Section 4's coordinate rule) — this is
        the only place in the whole system allowed to know a given model's
        native (tile-local) coordinate convention.
        """


def confidence_band(score: float) -> str:
    """Shared LOW/MEDIUM/HIGH banding so every adapter reports confidence consistently."""
    if score >= 0.75:
        return "HIGH"
    if score >= 0.4:
        return "MEDIUM"
    return "LOW"
