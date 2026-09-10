"""
Grounding adapter — Section 3.4: text-guided grounding (boxes and/or masks).

Runs per tile because grounding models operate at tile scale.

Every predicted box is converted from tile-local pixel coordinates into
full-image pixel coordinates using the tile's (col_off, row_off).

Overlapping tile detections are deduplicated with Part 3's NMS before
returning the final Evidence object.

Expected model interface:
    model.ground(image_path: str, query: str) ->
        [
            {
                "label": str,
                "box_px": [x0, y0, x1, y1],
                "mask_rle": str | None,
                "score": float
            },
            ...
        ]

The model returns boxes in TILE-LOCAL pixel coordinates.
This adapter converts them to FULL-IMAGE pixel coordinates.
"""

from __future__ import annotations

from typing import Any

from backend.model_registry.adapters.base import (
    BaseAdapter,
    confidence_band,
)
from backend.preprocessing.stitching import nms
from backend.shared.schemas import (
    Confidence,
    Detection,
    Evidence,
    Modality,
    ModelRegistryEntry,
    TaskType,
    Tile,
)


class GroundingAdapter(BaseAdapter):
    """Adapter for text-guided object grounding."""

    handles = (TaskType.GROUNDING,)

    def __init__(
        self,
        query: str,
        upstream_evidence=None,
    ):
        if not query or not query.strip():
            raise ValueError(
                "GroundingAdapter needs a non-empty text query to ground against."
            )

        self.query = query
        self.upstream_evidence = upstream_evidence or []

    def _upstream_context(self) -> str:
        """
        Convert useful upstream evidence into compact natural-language
        context for the grounding model.

        This keeps agent orchestration outside BaseAdapter while allowing
        downstream grounding to benefit from previous specialist results.
        """
        if not self.upstream_evidence:
            return ""

        parts: list[str] = []

        for evidence in self.upstream_evidence:
            parts.append(f"Previous task: {evidence.task.value}")

            if evidence.vqa_answer_raw:
                parts.append(
                    f"Previous answer: {evidence.vqa_answer_raw}"
                )

            if evidence.change_map is not None:
                parts.append(
                    "Changed area: "
                    f"{evidence.change_map.changed_area_px} px"
                )

            if evidence.detections:
                parts.append(
                    "Previous detections: "
                    f"{len(evidence.detections)}"
                )

        return "\n".join(parts)

    def preprocess(
        self,
        tiles: list[Tile],
        entry: ModelRegistryEntry,
    ) -> Any:
        """
        Prepare tile inputs for the grounding model.
        """
        return [
            {
                "tile": tile,
                "image_path": tile.array_path,
            }
            for tile in tiles
        ]

    def infer(
        self,
        model: Any,
        model_input: Any,
    ) -> Any:
        """
        Run grounding independently on every tile.
        """

        upstream_context = self._upstream_context()

        grounding_query = self.query

        if upstream_context:
            grounding_query = (
                f"{self.query}\n\n"
                "Additional evidence from previous analysis:\n"
                f"{upstream_context}"
            )

        results = []

        for item in model_input:
            results.append(
                {
                    "tile": item["tile"],
                    "raw": model.ground(
                        image_path=item["image_path"],
                        query=grounding_query,
                    ),
                }
            )

        return results

    def postprocess(
        self,
        raw_output: Any,
        tiles: list[Tile],
        entry: ModelRegistryEntry,
        modality_used: list[Modality],
    ) -> Evidence:
        """
        Convert tile-local detections to full-image coordinates,
        deduplicate overlapping detections, and create Evidence.
        """

        raw_detections: list[dict] = []

        for item in raw_output:
            tile: Tile = item["tile"]

            for detection in item["raw"]:
                x0, y0, x1, y1 = detection["box_px"]

                raw_detections.append(
                    {
                        "label": detection["label"],
                        "box_px": [
                            x0 + tile.col_off,
                            y0 + tile.row_off,
                            x1 + tile.col_off,
                            y1 + tile.row_off,
                        ],
                        "mask_rle": detection.get("mask_rle"),
                        "score": float(detection["score"]),
                    }
                )

        # Remove duplicate detections caused by overlapping tiles.
        deduped = nms(raw_detections)

        detections = [
            Detection(**detection)
            for detection in deduped
        ]

        mean_score = (
            sum(detection.score for detection in detections)
            / len(detections)
            if detections
            else 0.0
        )

        return Evidence(
            task=TaskType.GROUNDING,
            model_used=entry.name,
            modality_used=modality_used,
            detections=detections,
            confidence=Confidence(
                value=mean_score or None,
                band=confidence_band(mean_score),
                basis=(
                    f"mean of {len(detections)} detection score(s) "
                    "after tile-overlap NMS"
                    if detections
                    else "no detections above threshold"
                ),
            ),
            warnings=(
                []
                if detections
                else ["No matching objects found for this query."]
            ),
        )