"""
Grounding adapter — Section 3.4: text-guided grounding (boxes and/or masks).

Runs per tile (grounding models are tile-scale, not full-scene) and
canonicalizes every box to full-image pixel space by adding the tile's
(col_off, row_off) before returning — Section 4's coordinate rule.

Deduplicates boxes that straddle a tile boundary using Part 3's NMS
(backend.preprocessing.stitching.nms) before returning — added during the
Part 1-6 merge. Tiles overlap by design (tiling.py's default 15%
overlap_pct), so without this, the same real-world object detected in two
adjacent tiles would come back as two separate boxes in the merged
response. Uses `nms` directly rather than the full `stitch_detections`
(which also mosaics raster products this adapter doesn't produce).

Expected model interface:
    model.ground(image_path: str, query: str) ->
        [{"label": str, "box_px": [x0, y0, x1, y1], "mask_rle": str | None, "score": float}, ...]
    (box_px here is TILE-LOCAL pixel space; this adapter shifts it to
    full-image space in postprocess.)
"""
from __future__ import annotations

from typing import Any

from backend.model_registry.adapters.base import BaseAdapter, confidence_band
from backend.preprocessing.stitching import nms
from backend.shared.schemas import Confidence, Detection, Evidence, Modality, ModelRegistryEntry, TaskType, Tile


class GroundingAdapter(BaseAdapter):
    handles = (TaskType.GROUNDING,)

    def __init__(self, query: str):
        if not query or not query.strip():
            raise ValueError("GroundingAdapter needs a non-empty text query to ground against.")
        self.query = query

    def preprocess(self, tiles: list[Tile], entry: ModelRegistryEntry) -> Any:
        return [{"tile": t, "image_path": t.array_path} for t in tiles]

    def infer(self, model: Any, model_input: Any) -> Any:
        # One call per tile. A real batched-model implementation would fold
        # this into a single model.ground_batch(...) call for throughput —
        # that's an internal optimization this adapter is free to make later
        # without changing its preprocess/postprocess contract.
        return [
            {"tile": item["tile"], "raw": model.ground(image_path=item["image_path"], query=self.query)}
            for item in model_input
        ]

    def postprocess(
        self, raw_output: Any, tiles: list[Tile], entry: ModelRegistryEntry, modality_used: list[Modality],
    ) -> Evidence:
        raw_detections: list[dict] = []
        for item in raw_output:
            tile: Tile = item["tile"]
            for det in item["raw"]:
                x0, y0, x1, y1 = det["box_px"]
                raw_detections.append({
                    "label": det["label"],
                    "box_px": [x0 + tile.col_off, y0 + tile.row_off, x1 + tile.col_off, y1 + tile.row_off],
                    "mask_rle": det.get("mask_rle"),
                    "score": float(det["score"]),
                })

        deduped = nms(raw_detections)  # tile-overlap-boundary dedup, per-label, IoU-thresholded
        detections = [Detection(**d) for d in deduped]

        mean_score = sum(d.score for d in detections) / len(detections) if detections else 0.0
        return Evidence(
            task=TaskType.GROUNDING,
            model_used=entry.name,
            modality_used=modality_used,
            detections=detections,
            confidence=Confidence(
                value=mean_score or None,
                band=confidence_band(mean_score),
                basis=(
                    f"mean of {len(detections)} detection score(s) after tile-overlap NMS"
                    if detections else "no detections above threshold"
                ),
            ),
            warnings=[] if detections else ["No matching objects found for this query."],
        )
