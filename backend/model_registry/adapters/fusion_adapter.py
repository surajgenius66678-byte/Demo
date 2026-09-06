"""
Fusion adapter — Section 3.4: optical+SAR fusion.

Pairs tiles from the optical image with tiles from the SAR image at the
same spatial offset, then runs a joint model over each pair. Needs to know
which of the two input images is OPTICAL vs SAR — Tile itself carries no
modality field, see README.md#open-issues.

Expected model interface:
    model.fuse(optical_path: str, sar_path: str, prompt: str | None) -> {
        "text": str | None,
        "detections": [{"label": str, "box_px": [x0,y0,x1,y1], "mask_rle": str | None, "score": float}, ...],
    }
    (box_px here is TILE-LOCAL, same convention as the grounding adapter.)

Deduplicates boxes that straddle a tile boundary using Part 3's NMS
(backend.preprocessing.stitching.nms) before returning — same tile-overlap
reasoning and fix as grounding_adapter.py, added during the Part 1-6 merge.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from backend.model_registry.adapters.base import BaseAdapter, confidence_band
from backend.preprocessing.stitching import nms
from backend.shared.schemas import Confidence, Detection, Evidence, Modality, ModelRegistryEntry, TaskType, Tile


class FusionAdapter(BaseAdapter):
    handles = (TaskType.OPTICAL_SAR_FUSION,)

    def __init__(self, query: str | None = None, image_modalities: dict[str, Modality] | None = None):
        self.query = query
        self.image_modalities = image_modalities or {}

    def preprocess(self, tiles: list[Tile], entry: ModelRegistryEntry) -> Any:
        by_image: dict[str, list[Tile]] = defaultdict(list)
        for t in tiles:
            by_image[t.image_id].append(t)
        if len(by_image) != 2:
            raise ValueError(f"Fusion needs tiles from exactly 2 images (optical + SAR), got {len(by_image)}.")
        optical_id, sar_id = self._split_by_modality(list(by_image))
        pairs = self._pair_by_offset(by_image[optical_id], by_image[sar_id])
        return {"pairs": pairs, "prompt": self.query}

    def _split_by_modality(self, image_ids: list[str]) -> tuple[str, str]:
        labeled = {iid: self.image_modalities.get(iid) for iid in image_ids}
        optical = [iid for iid, m in labeled.items() if m == Modality.OPTICAL]
        sar = [iid for iid, m in labeled.items() if m == Modality.SAR]
        if len(optical) == 1 and len(sar) == 1:
            return optical[0], sar[0]
        # Best-effort fallback when modality wasn't provided — unreliable,
        # see README.md#open-issues. Assumes sorted image_id order.
        ids = sorted(image_ids)
        return ids[0], ids[1]

    @staticmethod
    def _pair_by_offset(a_tiles: list[Tile], b_tiles: list[Tile]) -> list[tuple[Tile, Tile]]:
        b_by_offset = {(t.col_off, t.row_off): t for t in b_tiles}
        return [(a, b_by_offset[(a.col_off, a.row_off)]) for a in a_tiles if (a.col_off, a.row_off) in b_by_offset]

    def infer(self, model: Any, model_input: Any) -> Any:
        return [
            {"optical": o, "sar": s, "raw": model.fuse(optical_path=o.array_path, sar_path=s.array_path, prompt=model_input["prompt"])}
            for o, s in model_input["pairs"]
        ]

    def postprocess(
        self, raw_output: Any, tiles: list[Tile], entry: ModelRegistryEntry, modality_used: list[Modality],
    ) -> Evidence:
        raw_detections: list[dict] = []
        texts: list[str] = []
        for item in raw_output:
            o = item["optical"]
            for det in item["raw"].get("detections", []):
                x0, y0, x1, y1 = det["box_px"]
                raw_detections.append({
                    "label": det["label"],
                    "box_px": [x0 + o.col_off, y0 + o.row_off, x1 + o.col_off, y1 + o.row_off],
                    "mask_rle": det.get("mask_rle"),
                    "score": float(det["score"]),
                })
            if item["raw"].get("text"):
                texts.append(item["raw"]["text"])

        deduped = nms(raw_detections)  # tile-overlap-boundary dedup, per-label, IoU-thresholded
        detections = [Detection(**d) for d in deduped]

        mean_score = sum(d.score for d in detections) / len(detections) if detections else 0.0
        return Evidence(
            task=TaskType.OPTICAL_SAR_FUSION,
            model_used=entry.name,
            modality_used=modality_used,
            detections=detections,
            vqa_answer_raw=" ".join(texts) if texts else None,
            confidence=Confidence(
                value=mean_score or None,
                band=confidence_band(mean_score),
                basis="mean fused-detection score after tile-overlap NMS",
            ),
            warnings=[] if (detections or texts) else ["Fusion model returned no detections or text for this pair."],
        )
