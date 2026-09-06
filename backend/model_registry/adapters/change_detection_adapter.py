"""
Change-detection adapter — Section 3.4: bi-temporal change detection +
change-based VQA.

Expects `tiles` to contain matched before/after pairs at the same spatial
offset (Part 3 tiles both images on one shared grid; Part 2 is responsible
for refusing this task upstream if check_coregistration reports
aligned=False, per its own Section 3.2 hardening list — by the time tiles
reach here they're assumed aligned).

Expected model interface:
    model.detect_change(before_path: str, after_path: str) -> {
        "probability_raster_path": str | None,
        "changed_area_px": int,
        "changed_area_pct": float,
        "mean_confidence": float,
    }
    model.answer(prompt: str, change_results: list[dict]) -> str   # optional, only for CHANGE_VQA
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from backend.model_registry.adapters.base import BaseAdapter, confidence_band
from backend.shared.schemas import ChangeMap, Confidence, Evidence, Modality, ModelRegistryEntry, TaskType, Tile


class ChangeDetectionAdapter(BaseAdapter):
    handles = (TaskType.CHANGE_DETECTION, TaskType.CHANGE_VQA)

    def __init__(self, query: str | None = None, image_order: list[str] | None = None):
        # query set -> CHANGE_VQA ("what changed near the river?"); unset -> CHANGE_DETECTION.
        # image_order: chronological [before_image_id, after_image_id]. Tile
        # itself carries no timestamp (see README.md#open-issues), so
        # without this the adapter falls back to sorting by image_id, which
        # is NOT guaranteed to mean "before then after" — flagged there too.
        self.query = query
        self.image_order = image_order

    def preprocess(self, tiles: list[Tile], entry: ModelRegistryEntry) -> Any:
        by_image: dict[str, list[Tile]] = defaultdict(list)
        for t in tiles:
            by_image[t.image_id].append(t)
        if len(by_image) != 2:
            raise ValueError(
                f"Change detection needs tiles from exactly 2 images (before/after), "
                f"got {len(by_image)}: {list(by_image)}"
            )
        ordered_ids = [iid for iid in (self.image_order or []) if iid in by_image]
        if len(ordered_ids) != 2:
            ordered_ids = sorted(by_image)  # best-effort fallback, see docstring above
        before_id, after_id = ordered_ids
        pairs = self._pair_by_offset(by_image[before_id], by_image[after_id])
        return {"pairs": pairs, "prompt": self.query}

    @staticmethod
    def _pair_by_offset(before_tiles: list[Tile], after_tiles: list[Tile]) -> list[tuple[Tile, Tile]]:
        after_by_offset = {(t.col_off, t.row_off): t for t in after_tiles}
        pairs = []
        for b in before_tiles:
            a = after_by_offset.get((b.col_off, b.row_off))
            if a is not None:
                pairs.append((b, a))
        return pairs

    def infer(self, model: Any, model_input: Any) -> Any:
        results = [
            {"before": b, "after": a, "raw": model.detect_change(before_path=b.array_path, after_path=a.array_path)}
            for b, a in model_input["pairs"]
        ]
        vqa_text = None
        if model_input["prompt"] and hasattr(model, "answer"):
            vqa_text = model.answer(prompt=model_input["prompt"], change_results=[r["raw"] for r in results])
        return {"tile_results": results, "vqa_text": vqa_text}

    def postprocess(
        self, raw_output: Any, tiles: list[Tile], entry: ModelRegistryEntry, modality_used: list[Modality],
    ) -> Evidence:
        tile_results = raw_output["tile_results"]
        task = TaskType.CHANGE_VQA if self.query else TaskType.CHANGE_DETECTION

        if not tile_results:
            return Evidence(
                task=task, model_used=entry.name, modality_used=modality_used,
                confidence=Confidence(value=None, band="LOW", basis="no spatially-matched tile pairs"),
                warnings=["No spatially-matched before/after tile pairs found."],
            )

        total_px = sum(r["raw"]["changed_area_px"] for r in tile_results)
        total_area_px = sum(r["before"].width * r["before"].height for r in tile_results) or 1
        mean_conf = sum(r["raw"]["mean_confidence"] for r in tile_results) / len(tile_results)
        change_map = ChangeMap(
            probability_raster_path=tile_results[0]["raw"].get("probability_raster_path"),
            changed_area_px=total_px,
            changed_area_pct=100.0 * total_px / total_area_px,
            mean_confidence=mean_conf,
        )
        return Evidence(
            task=task,
            model_used=entry.name,
            modality_used=modality_used,
            change_map=change_map,
            vqa_answer_raw=raw_output.get("vqa_text"),
            stats={"tile_pairs_compared": float(len(tile_results))},
            confidence=Confidence(
                value=mean_conf or None,
                band=confidence_band(mean_conf),
                basis="mean per-tile-pair change-model confidence",
            ),
            warnings=[],
        )
