"""
Part 4 — model loader.

Owns lazy GPU loading, LRU eviction under a VRAM budget, and quantized
loading (Section 3.4 hardening: "Lazy load + LRU evict — never every model
resident simultaneously by default"). Model construction itself is
delegated to a `model_factory` callable, so this file has zero hard
dependency on torch/transformers. That keeps the eviction/budget logic
unit-testable with a fake factory, and keeps this module importable in
environments where torch isn't installed yet — e.g. building Part 4 before
Part 6 has produced real checkpoints.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from backend.model_registry.exceptions import ModelLoadError
from backend.shared.schemas import ModelRegistryEntry

logger = logging.getLogger("satquery.model_registry.loader")

# (entry, device, quantization_override) -> loaded model handle.
# quantization_override lets run_inference's OOM ladder force a lighter
# load on retry without needing a second registry entry for the same model.
ModelFactory = Callable[[ModelRegistryEntry, str, Optional[str]], Any]


@dataclass
class LoadedModel:
    entry: ModelRegistryEntry
    handle: Any
    quantization: str
    vram_mb: float
    last_used: float = field(default_factory=time.monotonic)


def default_model_factory(entry: ModelRegistryEntry, device: str, quantization_override: str | None) -> Any:
    """
    Real loading path. Lazily imports torch so importing this module never
    requires it. This is the seam Part 6's checkpoint format plugs into —
    swap the body once Part 6 lands; ModelLoader's signature stays the same.
    """
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ModelLoadError(
            f"Cannot load '{entry.name}': torch is not installed in this "
            f"environment. Inject a custom model_factory, or use the mock "
            f"engine (configure_engine(use_mock=True)) until a GPU box with "
            f"requirements.txt installed is available."
        ) from exc

    # Placeholder body: real per-architecture loading (VLM / grounding /
    # change-detection / fusion) dispatches from here once Part 6's
    # checkpoint_metadata.json format is finalized (Section 3.6) and an
    # actual base model is chosen. quantization_override (or entry.quantization
    # if that's None) selects the load precision.
    raise ModelLoadError(
        f"default_model_factory has no real weights to load for "
        f"'{entry.name}' in this environment (checkpoint_path="
        f"{entry.checkpoint_path!r}). This is expected until Part 6 "
        f"delivers a checkpoint at that path — use the mock engine "
        f"until then."
    )


class ModelLoader:
    """
    Lazy-loads models on first use, evicts least-recently-used models when
    a new load would exceed `vram_budget_mb`, and never holds every model
    resident at once by default.
    """

    def __init__(
        self,
        registry: list[ModelRegistryEntry],
        vram_budget_mb: float = 8192.0,
        model_factory: ModelFactory = default_model_factory,
        device: str = "cuda",
    ):
        self.registry: dict[str, ModelRegistryEntry] = {e.name: e for e in registry}
        self.vram_budget_mb = vram_budget_mb
        self._model_factory = model_factory
        self.device = device
        self._resident: dict[str, LoadedModel] = {}

    def get(self, model_name: str, quantization_override: str | None = None) -> Any:
        """Return a ready-to-use model handle, loading (and evicting) as needed."""
        cached = self._resident.get(model_name)
        if cached is not None:
            if quantization_override is None or cached.quantization == quantization_override:
                cached.last_used = time.monotonic()
                return cached.handle
            # Same model, different quant tier requested (OOM retry ladder) —
            # drop the stale copy first so VRAM accounting never double-counts
            # one model resident under two quant tiers at once.
            self.unload(model_name)

        if model_name not in self.registry:
            raise ModelLoadError(f"'{model_name}' is not in the registry.")
        entry = self.registry[model_name]
        return self._load(entry, quantization_override).handle

    def _load(self, entry: ModelRegistryEntry, quantization_override: str | None) -> LoadedModel:
        handle = self._model_factory(entry, self.device, quantization_override)
        vram_mb = float(getattr(handle, "vram_mb", 0.0)) or _estimate_vram_mb(entry, quantization_override)

        self._evict_lru_until_fits(vram_mb)
        if not self._resident and self.vram_used_mb() + vram_mb > self.vram_budget_mb:
            logger.warning(
                "%s alone (%.0f MB) exceeds the %.0f MB budget; loading anyway (best effort).",
                entry.name, vram_mb, self.vram_budget_mb,
            )

        loaded = LoadedModel(
            entry=entry,
            handle=handle,
            quantization=quantization_override or entry.quantization,
            vram_mb=vram_mb,
        )
        self._resident[entry.name] = loaded
        logger.info(
            "Loaded %s (%s, %.0f MB); resident=%s (%.0f/%.0f MB)",
            entry.name, loaded.quantization, vram_mb,
            list(self._resident), self.vram_used_mb(), self.vram_budget_mb,
        )
        return loaded

    def _evict_lru_until_fits(self, incoming_vram_mb: float) -> None:
        while self._resident and self.vram_used_mb() + incoming_vram_mb > self.vram_budget_mb:
            victim_name = min(self._resident, key=lambda n: self._resident[n].last_used)
            self.unload(victim_name)

    def unload(self, model_name: str) -> None:
        loaded = self._resident.pop(model_name, None)
        if loaded is not None:
            close = getattr(loaded.handle, "close", None)
            if callable(close):
                close()
            logger.info("Evicted %s, freed %.0f MB", model_name, loaded.vram_mb)

    def resident_models(self) -> list[str]:
        return list(self._resident.keys())

    def vram_used_mb(self) -> float:
        return sum(m.vram_mb for m in self._resident.values())

    def health(self) -> dict:
        return {
            "models_loaded": self.resident_models(),
            "vram_used_mb": self.vram_used_mb(),
            "vram_budget_mb": self.vram_budget_mb,
        }


def _estimate_vram_mb(entry: ModelRegistryEntry, quantization_override: str | None) -> float:
    """Fallback footprint estimate when a factory doesn't report handle.vram_mb."""
    quant = quantization_override or entry.quantization
    base_mb = {"none": 6000.0, "8bit": 3200.0, "4bit": 1800.0}
    return base_mb.get(quant, 4000.0)
