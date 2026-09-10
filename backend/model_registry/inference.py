"""
Part 4 public interface — the exact functions Part 2 imports (Section 3.2):

    run_inference(task, tiles, model_hint=None) -> Evidence
    list_available_models() -> list[ModelRegistryEntry]
    health_check() -> dict

`query`, `image_modalities`, and `image_order` are additive keyword-only
extensions to the literal Section 3.2 signature — see README.md#open-issues
for why they exist (Tile carries neither modality nor timestamp, and the
spec's 3-arg run_inference has no field at all for the user's natural-
language query, which SINGLE_IMAGE_VQA / GROUNDING / CHANGE_VQA cannot
function without). Calling run_inference(task, tiles, model_hint) exactly
as Section 3.2 specifies still works — all three default to None and the
adapters degrade gracefully (a worse answer, not a crash) rather than
raising when they're omitted.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from backend.model_registry.adapters.base import BaseAdapter, confidence_band
from backend.model_registry.adapters.change_detection_adapter import ChangeDetectionAdapter
from backend.model_registry.adapters.fusion_adapter import FusionAdapter
from backend.model_registry.adapters.grounding_adapter import GroundingAdapter
from backend.model_registry.adapters.vlm_adapter import VLMAdapter
from backend.model_registry.loader import ModelLoader, default_model_factory
from backend.model_registry.registry import load_registry_config, select_model
from backend.shared.schemas import Confidence, Evidence, Modality, ModelRegistryEntry, TaskType, Tile

logger = logging.getLogger("satquery.model_registry.inference")


class _CallContext:
    __slots__ = (
        "query",
        "image_modalities",
        "image_order",
        "upstream_evidence",
    )

    def __init__(
        self,
        query,
        image_modalities,
        image_order,
        upstream_evidence=None,
    ):
        self.query = query
        self.image_modalities = image_modalities
        self.image_order = image_order
        self.upstream_evidence = upstream_evidence or []


_ADAPTER_FOR_TASK: dict[TaskType, Callable[[_CallContext], BaseAdapter]] = {
    TaskType.SINGLE_IMAGE_VQA: lambda ctx: VLMAdapter(query=ctx.query),
    TaskType.CAPTIONING: lambda ctx: VLMAdapter(query=None),
    TaskType.GROUNDING: lambda ctx: GroundingAdapter(query=ctx.query or ""),
    TaskType.CHANGE_DETECTION: lambda ctx: ChangeDetectionAdapter(query=None, image_order=ctx.image_order),
    TaskType.CHANGE_VQA: lambda ctx: ChangeDetectionAdapter(query=ctx.query, image_order=ctx.image_order),
    TaskType.OPTICAL_SAR_FUSION: lambda ctx: FusionAdapter(query=ctx.query, image_modalities=ctx.image_modalities),
}

# Quantization tiers tried in order before giving up on a given batch size —
# Section 3.4 hardening: "retry smaller batch -> smaller/quantized model ->
# explicit entry in Evidence.warnings, never a raw process crash."
_QUANT_FALLBACK: dict[str, str | None] = {"none": "8bit", "8bit": "4bit", "4bit": None}


def _is_oom(exc: Exception) -> bool:
    if isinstance(exc, MemoryError):
        return True
    # Real GPU code raises torch.cuda.OutOfMemoryError, a RuntimeError
    # subclass with "out of memory" in its message — this check catches it
    # without importing torch. MemoryError is the CPU-side analogue used by
    # tests/mocks so the ladder is exercisable without a GPU.
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _run_with_oom_ladder(
    tiles: list[Tile],
    entry: ModelRegistryEntry,
    process_batch: Callable[[list[Tile], str], Evidence],
) -> tuple[list[Evidence], list[str]]:
    """
    Shared OOM retry ladder: shrink batch size, then step down the
    quantization tier, then drop the offending tile — in that order, per
    Section 3.4. `process_batch(batch, quant)` does the real work for one
    batch at one quant tier; this function only owns the retry bookkeeping.
    Never raises for OOM (non-OOM exceptions propagate); always terminates,
    because `remaining` strictly shrinks every iteration (via `rest` on
    success or a dropped tile after all fallbacks are exhausted).
    """
    partial: list[Evidence] = []
    warnings: list[str] = []
    remaining = list(tiles)
    batch_size = len(remaining)
    quant = entry.quantization

    while remaining:
        batch, rest = remaining[:batch_size], remaining[batch_size:]
        try:
            partial.append(process_batch(batch, quant))
            remaining = rest
        except Exception as exc:
            if not _is_oom(exc):
                raise
            msg = f"OOM processing {len(batch)} tile(s) at quant={quant}: {exc}"
            logger.warning(msg)
            warnings.append(msg)
            if batch_size > 1:
                batch_size = max(1, batch_size // 2)
                continue
            next_quant = _QUANT_FALLBACK.get(quant)
            if next_quant is not None:
                quant = next_quant
                batch_size = len(remaining)
                continue
            warnings.append(f"Dropping tile {remaining[0].tile_id} after exhausting all OOM fallbacks.")
            remaining = remaining[1:]
            batch_size = len(remaining) or 1

    return partial, warnings


def _resolve_modality_used(
    tiles: list[Tile], entry: ModelRegistryEntry, image_modalities: dict[str, Modality] | None,
) -> list[Modality]:
    if image_modalities:
        found = sorted(
            {image_modalities[t.image_id] for t in tiles if t.image_id in image_modalities},
            key=lambda m: m.value,
        )
        if found:
            return found
    # Best-effort fallback — see README.md#open-issues (Tile has no modality field).
    return list(entry.modalities)


def _empty_evidence(task: TaskType, entry: ModelRegistryEntry, modality_used: list[Modality], warnings: list[str]) -> Evidence:
    return Evidence(
        task=task, model_used=entry.name, modality_used=modality_used,
        confidence=Confidence(value=None, band="LOW", basis="inference failed before producing evidence"),
        warnings=warnings,
    )


def _merge_evidences(evidences: list[Evidence], task: TaskType, entry: ModelRegistryEntry, modality_used: list[Modality]) -> Evidence:
    from backend.shared.schemas import ChangeMap

    all_detections = [d for e in evidences for d in e.detections]
    texts = [e.vqa_answer_raw for e in evidences if e.vqa_answer_raw]
    change_maps = [e.change_map for e in evidences if e.change_map is not None]

    merged_stats: dict[str, float] = {}
    for e in evidences:
        for k, v in e.stats.items():
            merged_stats[k] = merged_stats.get(k, 0.0) + v

    values = [e.confidence.value for e in evidences if e.confidence.value is not None]
    mean_value = sum(values) / len(values) if values else None

    merged_change_map = None
    if change_maps:
        merged_change_map = ChangeMap(
            probability_raster_path=change_maps[0].probability_raster_path,
            changed_area_px=sum(cm.changed_area_px for cm in change_maps),
            changed_area_pct=sum(cm.changed_area_pct for cm in change_maps) / len(change_maps),
            mean_confidence=sum(cm.mean_confidence for cm in change_maps) / len(change_maps),
        )

    return Evidence(
        task=task,
        model_used=entry.name,
        modality_used=modality_used,
        detections=all_detections,
        change_map=merged_change_map,
        vqa_answer_raw=" ".join(texts) if texts else None,
        stats=merged_stats,
        confidence=Confidence(
            value=mean_value,
            band=confidence_band(mean_value or 0.0),
            basis=f"merged across {len(evidences)} batch(es)",
        ),
        warnings=[w for e in evidences for w in e.warnings],
    )


class InferenceEngine:
    def __init__(self, registry: list[ModelRegistryEntry], loader: ModelLoader | None = None):
        self.registry = registry
        self.loader = loader or ModelLoader(registry)

    def run_inference(
        self,
        task: TaskType,
        tiles: list[Tile],
        model_hint: str | None = None,
        *,
        query: str | None = None,
        image_modalities: dict[str, Modality] | None = None,
        image_order: list[str] | None = None,
        upstream_evidence: list[Evidence] | None = None,
    ) -> Evidence:
        if task not in _ADAPTER_FOR_TASK:
            raise ValueError(
                f"Part 4 has no adapter for task={task!r} — route it to UNSUPPORTED in Part 2's planner instead."
            )
        if not tiles:
            raise ValueError("run_inference requires at least one tile.")

        ctx = _CallContext(query, image_modalities, image_order ,upstream_evidence,)
        adapter = _ADAPTER_FOR_TASK[task](ctx)
        entry = select_model(
            self.registry, task,
            modalities=list(image_modalities.values()) if image_modalities else None,
            model_hint=model_hint,
        )
        modality_used = _resolve_modality_used(tiles, entry, image_modalities)

        def process_batch(batch: list[Tile], quant: str) -> Evidence:
            model = self.loader.get(entry.name, quantization_override=(quant if quant != entry.quantization else None))
            model_input = adapter.preprocess(batch, entry)
            raw_output = adapter.infer(model, model_input)
            return adapter.postprocess(raw_output, batch, entry, modality_used)

        partial_evidences, oom_warnings = _run_with_oom_ladder(tiles, entry, process_batch)

        if not partial_evidences:
            return _empty_evidence(task, entry, modality_used, oom_warnings or ["No tiles could be processed."])

        merged = _merge_evidences(partial_evidences, task, entry, modality_used)
        merged.warnings = [*oom_warnings, *merged.warnings]
        return merged

    def list_available_models(self) -> list[ModelRegistryEntry]:
        return list(self.registry)

    def health_check(self) -> dict:
        return {"status": "ok", **self.loader.health()}


# --- Module-level singleton + Section 3.2's exact bare-function contract ---

_engine: InferenceEngine | Any = None


def configure_engine(
    use_mock: bool = False,
    registry_path: str | None = None,
    vram_budget_mb: float = 8192.0,
    **mock_kwargs: Any,
) -> Any:
    """Call once at process startup. use_mock=True is what Parts 2 and 5 want today."""
    global _engine
    if use_mock:
        from backend.model_registry.mock_registry import build_mock_engine
        _engine = build_mock_engine(**mock_kwargs)
    else:
        registry = load_registry_config(registry_path) if registry_path else load_registry_config()
        _engine = InferenceEngine(registry, ModelLoader(registry, vram_budget_mb=vram_budget_mb, model_factory=default_model_factory))
    return _engine


def get_engine() -> Any:
    if _engine is None:
        configure_engine(use_mock=True)  # safe default so importing this module never hard-fails without a GPU
    return _engine


def run_inference(task: TaskType, tiles: list[Tile], model_hint: str | None = None, **kwargs: Any) -> Evidence:
    return get_engine().run_inference(task, tiles, model_hint, **kwargs)


def list_available_models() -> list[ModelRegistryEntry]:
    return get_engine().list_available_models()


def health_check() -> dict:
    return get_engine().health_check()
