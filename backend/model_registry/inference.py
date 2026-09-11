"""
Part 4 public interface — the exact functions Part 2 imports (Section 3.2):

    run_inference(task, tiles, model_hint=None) -> Evidence
    list_available_models() -> list[ModelRegistryEntry]
    health_check() -> dict

Additional keyword-only arguments:
    query
    image_modalities
    image_order
    upstream_evidence

These are needed because Tile does not contain the natural-language query,
modality, or temporal ordering information required by several specialist
tasks.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from backend.model_registry.adapters.base import BaseAdapter, confidence_band
from backend.model_registry.adapters.change_detection_adapter import (
    ChangeDetectionAdapter,
)
from backend.model_registry.adapters.fusion_adapter import FusionAdapter
from backend.model_registry.adapters.grounding_adapter import GroundingAdapter
from backend.model_registry.adapters.vlm_adapter import VLMAdapter
from backend.model_registry.loader import (
    ModelLoader,
    default_model_factory,
)
from backend.model_registry.registry import (
    load_registry_config,
    select_model,
)
from backend.shared.schemas import (
    ChangeMap,
    Confidence,
    Evidence,
    Modality,
    ModelRegistryEntry,
    TaskType,
    Tile,
)

logger = logging.getLogger("satquery.model_registry.inference")


# ---------------------------------------------------------------------------
# Built-in specialist factory registration
# ---------------------------------------------------------------------------
#
# The OSCD change detector is a real specialist model. Register it here,
# lazily, so importing inference.py does not create circular imports.
#
# Other specialists such as grounding/fusion may still be placeholders until
# their actual model implementations are added.
# ---------------------------------------------------------------------------

_SPECIALIST_FACTORIES_REGISTERED = False


def _ensure_specialist_factories() -> None:
    """
    Register built-in specialist model factories exactly once.
    """
    global _SPECIALIST_FACTORIES_REGISTERED

    if _SPECIALIST_FACTORIES_REGISTERED:
        return

    try:
        from backend.model_registry.loader import register_model_factory
        from backend.model_registry.specialists.change_detector_factory import (
            load_oscd_change_detector,
        )

        register_model_factory(
            "satquery-change-bitemporal",
            load_oscd_change_detector,
        )

        logger.info(
            "Registered built-in specialist factory: satquery-change-bitemporal"
        )

    except ImportError as exc:
        # Do not make module import fail merely because an optional specialist
        # dependency is unavailable.
        logger.warning(
            "Could not register built-in specialist factories: %s",
            exc,
        )

    _SPECIALIST_FACTORIES_REGISTERED = True


# ---------------------------------------------------------------------------
# Adapter routing
# ---------------------------------------------------------------------------


class _CallContext:
    """
    Runtime information required by adapters but not stored in Tile.
    """

    __slots__ = (
        "query",
        "image_modalities",
        "image_order",
        "upstream_evidence",
    )

    def __init__(
        self,
        query: str | None,
        image_modalities: dict[str, Modality] | None,
        image_order: list[str] | None,
        upstream_evidence: list[Evidence] | None = None,
    ) -> None:
        self.query = query
        self.image_modalities = image_modalities
        self.image_order = image_order
        self.upstream_evidence = upstream_evidence or []


_ADAPTER_FOR_TASK: dict[
    TaskType,
    Callable[[_CallContext], BaseAdapter],
] = {
    TaskType.SINGLE_IMAGE_VQA: lambda ctx: VLMAdapter(
        query=ctx.query,
    ),

    TaskType.CAPTIONING: lambda ctx: VLMAdapter(
        query=None,
    ),

    TaskType.GROUNDING: lambda ctx: GroundingAdapter(
        query=ctx.query or "",
        upstream_evidence=ctx.upstream_evidence,
    ),

    TaskType.CHANGE_DETECTION: lambda ctx: ChangeDetectionAdapter(
        query=None,
        image_order=ctx.image_order,
    ),

    TaskType.CHANGE_VQA: lambda ctx: ChangeDetectionAdapter(
        query=ctx.query,
        image_order=ctx.image_order,
    ),

    TaskType.OPTICAL_SAR_FUSION: lambda ctx: FusionAdapter(
        query=ctx.query,
        image_modalities=ctx.image_modalities,
    ),
}


# ---------------------------------------------------------------------------
# OOM fallback
# ---------------------------------------------------------------------------
#
# Section 3.4 hardening:
#
#   1. Reduce batch size
#   2. Try smaller quantization tier
#   3. Drop offending tile if all fallbacks fail
#
# OOM is converted into Evidence warnings instead of crashing the whole
# inference pipeline.
# ---------------------------------------------------------------------------

_QUANT_FALLBACK: dict[str, str | None] = {
    "none": "8bit",
    "8bit": "4bit",
    "4bit": None,
}


def _is_oom(exc: Exception) -> bool:
    """
    Return True when an exception represents an out-of-memory condition.
    """

    if isinstance(exc, MemoryError):
        return True

    return (
        isinstance(exc, RuntimeError)
        and "out of memory" in str(exc).lower()
    )


def _run_with_oom_ladder(
    tiles: list[Tile],
    entry: ModelRegistryEntry,
    process_batch: Callable[[list[Tile], str], Evidence],
) -> tuple[list[Evidence], list[str]]:
    """
    Execute inference with the shared OOM fallback ladder.

    Order:

        batch size reduction
            ↓
        quantization fallback
            ↓
        drop tile

    Non-OOM exceptions are allowed to propagate because they represent real
    programming/model/data errors rather than recoverable memory pressure.
    """

    partial: list[Evidence] = []
    warnings: list[str] = []

    remaining = list(tiles)

    batch_size = len(remaining)
    quant = entry.quantization

    while remaining:
        batch = remaining[:batch_size]
        rest = remaining[batch_size:]

        try:
            evidence = process_batch(batch, quant)

            partial.append(evidence)
            remaining = rest

        except Exception as exc:
            if not _is_oom(exc):
                raise

            message = (
                f"OOM processing {len(batch)} tile(s) "
                f"at quant={quant}: {exc}"
            )

            logger.warning(message)
            warnings.append(message)

            # ---------------------------------------------------------------
            # First fallback: smaller batch
            # ---------------------------------------------------------------

            if batch_size > 1:
                new_batch_size = max(1, batch_size // 2)

                logger.warning(
                    "Reducing inference batch size: %d -> %d",
                    batch_size,
                    new_batch_size,
                )

                batch_size = new_batch_size
                continue

            # ---------------------------------------------------------------
            # Second fallback: lower quantization
            # ---------------------------------------------------------------

            next_quant = _QUANT_FALLBACK.get(quant)

            if next_quant is not None:
                logger.warning(
                    "Changing quantization fallback: %s -> %s",
                    quant,
                    next_quant,
                )

                quant = next_quant
                batch_size = len(remaining)

                continue

            # ---------------------------------------------------------------
            # Final fallback: drop offending tile
            # ---------------------------------------------------------------

            dropped_tile = remaining[0]

            drop_message = (
                f"Dropping tile {dropped_tile.tile_id} "
                "after exhausting all OOM fallbacks."
            )

            logger.warning(drop_message)
            warnings.append(drop_message)

            remaining = remaining[1:]

            batch_size = len(remaining) or 1

    return partial, warnings


# ---------------------------------------------------------------------------
# Modality handling
# ---------------------------------------------------------------------------


def _resolve_modality_used(
    tiles: list[Tile],
    entry: ModelRegistryEntry,
    image_modalities: dict[str, Modality] | None,
) -> list[Modality]:
    """
    Determine which modalities were actually used.

    Tile itself does not contain modality information, so image_modalities is
    preferred when available. Otherwise we fall back to the modalities
    declared by the selected model registry entry.
    """

    if image_modalities:
        found = sorted(
            {
                image_modalities[tile.image_id]
                for tile in tiles
                if tile.image_id in image_modalities
            },
            key=lambda modality: modality.value,
        )

        if found:
            return found

    return list(entry.modalities)


# ---------------------------------------------------------------------------
# Evidence helpers
# ---------------------------------------------------------------------------


def _empty_evidence(
    task: TaskType,
    entry: ModelRegistryEntry,
    modality_used: list[Modality],
    warnings: list[str],
) -> Evidence:
    """
    Create an Evidence object when no successful inference was produced.
    """

    return Evidence(
        task=task,
        model_used=entry.name,
        modality_used=modality_used,
        confidence=Confidence(
            value=None,
            band="LOW",
            basis="inference failed before producing evidence",
        ),
        warnings=warnings,
    )


def _merge_evidences(
    evidences: list[Evidence],
    task: TaskType,
    entry: ModelRegistryEntry,
    modality_used: list[Modality],
) -> Evidence:
    """
    Merge evidence produced by multiple inference batches.
    """

    all_detections = [
        detection
        for evidence in evidences
        for detection in evidence.detections
    ]

    text_answers = [
        evidence.vqa_answer_raw
        for evidence in evidences
        if evidence.vqa_answer_raw
    ]

    change_maps = [
        evidence.change_map
        for evidence in evidences
        if evidence.change_map is not None
    ]

    # -----------------------------------------------------------------------
    # Merge numeric statistics
    # -----------------------------------------------------------------------

    merged_stats: dict[str, float] = {}

    for evidence in evidences:
        for key, value in evidence.stats.items():
            merged_stats[key] = (
                merged_stats.get(key, 0.0) + value
            )

    # -----------------------------------------------------------------------
    # Merge confidence
    # -----------------------------------------------------------------------

    confidence_values = [
        evidence.confidence.value
        for evidence in evidences
        if evidence.confidence.value is not None
    ]

    mean_confidence = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values
        else None
    )

    # -----------------------------------------------------------------------
    # Merge change maps
    # -----------------------------------------------------------------------

    merged_change_map: ChangeMap | None = None

    if change_maps:
        merged_change_map = ChangeMap(
            probability_raster_path=(
                change_maps[0].probability_raster_path
            ),
            changed_area_px=sum(
                change_map.changed_area_px
                for change_map in change_maps
            ),
            changed_area_pct=(
                sum(
                    change_map.changed_area_pct
                    for change_map in change_maps
                )
                / len(change_maps)
            ),
            mean_confidence=(
                sum(
                    change_map.mean_confidence
                    for change_map in change_maps
                )
                / len(change_maps)
            ),
        )

    # -----------------------------------------------------------------------
    # Merge warnings
    # -----------------------------------------------------------------------

    merged_warnings = [
        warning
        for evidence in evidences
        for warning in evidence.warnings
    ]

    return Evidence(
        task=task,
        model_used=entry.name,
        modality_used=modality_used,
        detections=all_detections,
        change_map=merged_change_map,
        vqa_answer_raw=(
            " ".join(text_answers)
            if text_answers
            else None
        ),
        stats=merged_stats,
        confidence=Confidence(
            value=mean_confidence,
            band=confidence_band(mean_confidence or 0.0),
            basis=(
                f"merged across {len(evidences)} batch(es)"
            ),
        ),
        warnings=merged_warnings,
    )


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------


class InferenceEngine:
    """
    Part 4 inference engine.

    Responsibilities:

        1. Select appropriate model from registry.
        2. Select adapter for requested task.
        3. Preprocess tiles.
        4. Load model through ModelLoader.
        5. Execute inference.
        6. Postprocess into Evidence.
        7. Handle recoverable OOM conditions.
    """

    def __init__(
        self,
        registry: list[ModelRegistryEntry],
        loader: ModelLoader | None = None,
    ) -> None:
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
        """
        Run a complete Part 4 inference request.
        """

        # Make sure built-in specialist factories are available.
        _ensure_specialist_factories()

        # -------------------------------------------------------------------
        # Validate task
        # -------------------------------------------------------------------

        if task not in _ADAPTER_FOR_TASK:
            raise ValueError(
                f"Part 4 has no adapter for task={task!r} — "
                "route it to UNSUPPORTED in Part 2's planner instead."
            )

        # -------------------------------------------------------------------
        # Validate tiles
        # -------------------------------------------------------------------

        if not tiles:
            raise ValueError(
                "run_inference requires at least one tile."
            )

        # -------------------------------------------------------------------
        # Build adapter context
        # -------------------------------------------------------------------

        context = _CallContext(
            query=query,
            image_modalities=image_modalities,
            image_order=image_order,
            upstream_evidence=upstream_evidence,
        )

        adapter = _ADAPTER_FOR_TASK[task](context)

        # -------------------------------------------------------------------
        # Select model
        # -------------------------------------------------------------------

        selected_modalities = (
            list(image_modalities.values())
            if image_modalities
            else None
        )

        entry = select_model(
            self.registry,
            task,
            modalities=selected_modalities,
            model_hint=model_hint,
        )

        modality_used = _resolve_modality_used(
            tiles,
            entry,
            image_modalities,
        )

        logger.info(
            "Selected model=%s for task=%s",
            entry.name,
            task.value,
        )

        # -------------------------------------------------------------------
        # Process one batch
        # -------------------------------------------------------------------

        def process_batch(
            batch: list[Tile],
            quantization: str,
        ) -> Evidence:
            """
            Execute one adapter batch.
            """

            quantization_override = (
                quantization
                if quantization != entry.quantization
                else None
            )

            model = self.loader.get(
                entry.name,
                quantization_override=quantization_override,
            )

            model_input = adapter.preprocess(
                batch,
                entry,
            )

            raw_output = adapter.infer(
                model,
                model_input,
            )

            return adapter.postprocess(
                raw_output,
                batch,
                entry,
                modality_used,
            )

        # -------------------------------------------------------------------
        # Execute with OOM fallback
        # -------------------------------------------------------------------

        partial_evidences, oom_warnings = _run_with_oom_ladder(
            tiles,
            entry,
            process_batch,
        )

        # -------------------------------------------------------------------
        # Nothing succeeded
        # -------------------------------------------------------------------

        if not partial_evidences:
            return _empty_evidence(
                task=task,
                entry=entry,
                modality_used=modality_used,
                warnings=(
                    oom_warnings
                    or ["No tiles could be processed."]
                ),
            )

        # -------------------------------------------------------------------
        # Merge results
        # -------------------------------------------------------------------

        merged = _merge_evidences(
            evidences=partial_evidences,
            task=task,
            entry=entry,
            modality_used=modality_used,
        )

        # OOM warnings should appear first so they are easy to audit.
        merged.warnings = [
            *oom_warnings,
            *merged.warnings,
        ]

        return merged

    # -----------------------------------------------------------------------
    # Public helpers
    # -----------------------------------------------------------------------

    def list_available_models(
        self,
    ) -> list[ModelRegistryEntry]:
        return list(self.registry)

    def health_check(self) -> dict:
        return {
            "status": "ok",
            **self.loader.health(),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_engine: InferenceEngine | Any = None


def configure_engine(
    use_mock: bool = False,
    registry_path: str | None = None,
    vram_budget_mb: float = 8192.0,
    **mock_kwargs: Any,
) -> Any:
    """
    Configure the process-wide inference engine.

    During development/tests:
        configure_engine(use_mock=True)

    Real inference:
        configure_engine(use_mock=False)
    """

    global _engine

    if use_mock:
        from backend.model_registry.mock_registry import (
            build_mock_engine,
        )

        _engine = build_mock_engine(
            **mock_kwargs,
        )

        return _engine

    # -----------------------------------------------------------------------
    # Real registry
    # -----------------------------------------------------------------------

    registry = (
        load_registry_config(registry_path)
        if registry_path
        else load_registry_config()
    )

    _ensure_specialist_factories()

    loader = ModelLoader(
        registry,
        vram_budget_mb=vram_budget_mb,
        model_factory=default_model_factory,
    )

    _engine = InferenceEngine(
        registry=registry,
        loader=loader,
    )

    return _engine


def get_engine() -> Any:
    """
    Return the configured inference engine.

    If nothing has been configured yet, use the mock engine. This keeps
    importing the module safe on machines without GPU/model dependencies.
    """

    if _engine is None:
        configure_engine(
            use_mock=True,
        )

    return _engine


# ---------------------------------------------------------------------------
# Section 3.2 public functions
# ---------------------------------------------------------------------------


def run_inference(
    task: TaskType,
    tiles: list[Tile],
    model_hint: str | None = None,
    **kwargs: Any,
) -> Evidence:
    """
    Public Part 4 inference entry point.
    """

    return get_engine().run_inference(
        task,
        tiles,
        model_hint,
        **kwargs,
    )


def list_available_models() -> list[ModelRegistryEntry]:
    """
    Return all models known to the active registry.
    """

    return get_engine().list_available_models()


def health_check() -> dict:
    """
    Return Part 4/model-loader health information.
    """

    return get_engine().health_check()