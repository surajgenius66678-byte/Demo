"""
stitch_detections() — architecture.md Section 3.3:
  "the global stitching/NMS helper used after Part 4 returns per-tile results"

Overlapping tiles independently detect the same real-world object, so the
same object shows up as near-duplicate detections in the merged, full-image
coordinate space (every box arrives already in full-image pixel space per
the coordinate rule in Section 4 — Part 4's adapters canonicalize that
before returning). Stitching's job is to deduplicate those via NMS and merge
the raster-based products (change maps) into one mosaic.

Split the same way as tiling.py / coregistration.py:
  - box_iou / mask_iou / nms / encode_mask_rle / decode_mask_rle /
    merge_stats / merge_change_maps / derive_confidence: pure, numpy-only,
    fully unit-testable.
  - stitch_detections: the literal public interface, pydantic + rasterio
    dependent (raster mosaicking), not executable in the build sandbox.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import config

if TYPE_CHECKING:
    from backend.shared.schemas import Evidence


# --------------------------------------------------------------------------
# Box / mask IoU
# --------------------------------------------------------------------------

def box_iou(box_a: list[float], box_b: list[float]) -> float:
    """IoU of two [x0, y0, x1, y1] boxes. Degenerate (zero-area) boxes
    return 0.0 rather than raising or producing NaN."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b

    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """IoU of two same-shape boolean masks."""
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"masks must be the same shape to compare, got {mask_a.shape} vs {mask_b.shape}")
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    inter = np.logical_and(a, b).sum()
    return float(inter) / float(union)


def _detection_iou(det_a: dict, det_b: dict) -> float:
    """Uses mask IoU when both detections carry a mask_rle (more accurate
    for instance-segmentation-style dedup); falls back to box IoU otherwise
    (e.g. pure bounding-box grounding results, no masks)."""
    rle_a, rle_b = det_a.get("mask_rle"), det_b.get("mask_rle")
    if rle_a and rle_b:
        mask_a, mask_b = decode_mask_rle(rle_a), decode_mask_rle(rle_b)
        if mask_a.shape == mask_b.shape:
            return mask_iou(mask_a, mask_b)
    return box_iou(det_a["box_px"], det_b["box_px"])


# --------------------------------------------------------------------------
# Mask RLE — simple custom format, documented since the schema doesn't pin
# one down (mask_rle: Optional[str])
# --------------------------------------------------------------------------

def encode_mask_rle(mask: np.ndarray) -> str:
    """
    Row-major (C-order) binary run-length encoding.
    Format: "{height}x{width}:{c0},{c1},{c2},..." — counts alternate runs
    starting with a run of 0s (background); if the mask starts with a 1,
    the first count is 0 (an empty leading background run), so decode
    always knows which value the first real run represents.
    """
    if mask.ndim != 2:
        raise ValueError(f"expected a 2D mask, got shape {mask.shape}")
    flat = mask.astype(np.uint8).ravel(order="C")
    header = f"{mask.shape[0]}x{mask.shape[1]}"
    if flat.size == 0:
        return f"{header}:"

    change_points = np.where(np.diff(flat) != 0)[0] + 1
    run_starts = np.concatenate(([0], change_points))
    run_ends = np.concatenate((change_points, [flat.size]))
    run_lengths = (run_ends - run_starts).tolist()

    if flat[0] == 1:
        run_lengths = [0] + run_lengths

    return header + ":" + ",".join(str(int(c)) for c in run_lengths)


def decode_mask_rle(rle: str) -> np.ndarray:
    """Inverse of encode_mask_rle."""
    header, _, counts_str = rle.partition(":")
    h_str, w_str = header.split("x")
    h, w = int(h_str), int(w_str)

    flat = np.zeros(h * w, dtype=np.uint8)
    if counts_str:
        pos = 0
        value = 0
        for count in (int(c) for c in counts_str.split(",")):
            if value == 1:
                flat[pos : pos + count] = 1
            pos += count
            value = 1 - value
    return flat.reshape((h, w)).astype(bool)


# --------------------------------------------------------------------------
# Non-max suppression
# --------------------------------------------------------------------------

def nms(detections: list[dict], iou_threshold: float | None = None) -> list[dict]:
    """
    Greedy NMS, applied independently within each label (a "building" box
    never suppresses a "road" box, even if heavily overlapping). Detections
    are ranked by score descending; a lower-scoring detection is dropped if
    its overlap with an already-kept detection of the same label strictly
    exceeds iou_threshold — an overlap exactly equal to the threshold is
    kept, not suppressed.
    """
    if iou_threshold is None:
        iou_threshold = config.STITCH_IOU_THRESHOLD
    if not detections:
        return []

    by_label: dict[str, list[dict]] = {}
    for det in detections:
        by_label.setdefault(det["label"], []).append(det)

    kept: list[dict] = []
    for _label, dets in by_label.items():
        ranked = sorted(dets, key=lambda d: d["score"], reverse=True)
        suppressed = [False] * len(ranked)
        for i, det_i in enumerate(ranked):
            if suppressed[i]:
                continue
            kept.append(det_i)
            for j in range(i + 1, len(ranked)):
                if suppressed[j]:
                    continue
                if _detection_iou(det_i, ranked[j]) > iou_threshold:
                    suppressed[j] = True
    return kept


# --------------------------------------------------------------------------
# Merging stats / change maps / confidence
# --------------------------------------------------------------------------

def merge_stats(stats_list: list[dict]) -> dict:
    """
    Merges per-tile `stats` dicts by averaging values sharing a key.
    Evidence.stats carries no per-key semantics (it's a free-form
    dict[str, float]), so this is a deliberately simple, documented default
    — a caller with sum-type stats (e.g. an object count) should aggregate
    those upstream with key-specific logic; this generic merge always
    averages.
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for stats in stats_list:
        for key, value in stats.items():
            sums[key] = sums.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sums}


def merge_change_maps(change_maps: list[dict], total_image_px: int | None = None) -> dict:
    """
    Pure aggregation across per-tile change_map dicts (each shaped like
    ChangeMap: changed_area_px, changed_area_pct, mean_confidence,
    probability_raster_path).

    changed_area_px is summed — note this can double-count pixels that fall
    in more than one tile's overlap margin. stitch_detections() corrects
    this when raster paths are available, by mosaicking the actual rasters
    and recomputing both figures from the merged array instead of trusting
    this sum; this function is the fallback for when no rasters are
    available to mosaic (or as the basis before that correction runs).

    mean_confidence is weighted by each tile's changed_area_px (tiles with
    more detected change contribute proportionally more), falling back to a
    plain mean if every tile reports zero changed area.
    """
    if not change_maps:
        return {
            "probability_raster_path": None,
            "changed_area_px": 0,
            "changed_area_pct": 0.0,
            "mean_confidence": 0.0,
        }

    total_changed_px = sum(cm["changed_area_px"] for cm in change_maps)

    weight_sum = sum(cm["changed_area_px"] for cm in change_maps)
    if weight_sum > 0:
        mean_confidence = sum(
            cm["mean_confidence"] * cm["changed_area_px"] for cm in change_maps
        ) / weight_sum
    else:
        mean_confidence = sum(cm["mean_confidence"] for cm in change_maps) / len(change_maps)

    if total_image_px:
        changed_area_pct = 100.0 * total_changed_px / total_image_px
    else:
        changed_area_pct = sum(cm["changed_area_pct"] for cm in change_maps) / len(change_maps)

    return {
        "probability_raster_path": None,  # filled in by stitch_detections if a mosaic is written
        "changed_area_px": int(total_changed_px),
        "changed_area_pct": changed_area_pct,
        "mean_confidence": mean_confidence,
    }


def derive_confidence(values: list[float], basis: str) -> dict:
    """
    Deterministic confidence banding from a list of real scores already
    present in the evidence being merged — never an invented number.
    Thresholds are a simple, documented default (matches the spirit of
    Part 5's confidence rule, which Part 3 also needs to satisfy here since
    stitch_detections must return a schema-valid Evidence with *a*
    Confidence, ahead of Part 5's own final-response confidence pass).
    """
    if not values:
        return {"value": None, "band": "LOW", "basis": "no detections or change signal to derive confidence from"}
    mean_value = sum(values) / len(values)
    if mean_value < 0.4:
        band = "LOW"
    elif mean_value < 0.7:
        band = "MEDIUM"
    else:
        band = "HIGH"
    return {"value": mean_value, "band": band, "basis": basis}


# --------------------------------------------------------------------------
# I/O layer — rasterio + pydantic dependent, not executable in the build
# sandbox
# --------------------------------------------------------------------------

def stitch_detections(tile_evidence: "list[Evidence]", image_id: str) -> "Evidence":
    """
    architecture.md 3.3 literal interface:
        def stitch_detections(tile_evidence, image_id) -> Evidence
    """
    from backend.shared.schemas import Evidence, Detection, ChangeMap, Confidence
    from . import store

    if not tile_evidence:
        raise ValueError("tile_evidence must be non-empty")

    tasks = {ev.task for ev in tile_evidence}
    if len(tasks) > 1:
        raise ValueError(f"stitch_detections received mixed task types across tiles: {tasks}")
    task = next(iter(tasks))

    models_used = {ev.model_used for ev in tile_evidence}
    model_used = models_used.pop() if len(models_used) == 1 else "+".join(sorted(models_used))

    modality_used: list = []
    for ev in tile_evidence:
        for m in ev.modality_used:
            if m not in modality_used:
                modality_used.append(m)

    all_detections = [d.model_dump() for ev in tile_evidence for d in ev.detections]
    kept_detections = nms(all_detections)

    warnings: list[str] = []
    for ev in tile_evidence:
        for w in ev.warnings:
            if w not in warnings:
                warnings.append(w)

    change_map_dicts = [ev.change_map.model_dump() for ev in tile_evidence if ev.change_map is not None]
    merged_change_map = None
    if change_map_dicts:
        meta = store.load_metadata(image_id)
        total_px = meta["width"] * meta["height"]
        merged = merge_change_maps(change_map_dicts, total_image_px=total_px)

        raster_paths = [cm["probability_raster_path"] for cm in change_map_dicts]
        if all(raster_paths):
            mosaic_result = _mosaic_and_recount(raster_paths, image_id, total_px)
            if mosaic_result is not None:
                merged["probability_raster_path"] = mosaic_result["probability_raster_path"]
                merged["changed_area_px"] = mosaic_result["changed_area_px"]
                merged["changed_area_pct"] = mosaic_result["changed_area_pct"]

        merged_change_map = ChangeMap(**merged)

    vqa_answers = [ev.vqa_answer_raw for ev in tile_evidence if ev.vqa_answer_raw]
    vqa_answer_raw = vqa_answers[0] if vqa_answers else None
    if len(set(vqa_answers)) > 1:
        warnings.append(
            f"stitch_detections received {len(set(vqa_answers))} distinct vqa_answer_raw values "
            f"across tiles; kept the first and discarded the rest"
        )

    merged_stats = merge_stats([ev.stats for ev in tile_evidence])

    if kept_detections:
        confidence = derive_confidence(
            [d["score"] for d in kept_detections],
            basis=f"mean of {len(kept_detections)} post-NMS detection scores across {len(tile_evidence)} tiles",
        )
    elif merged_change_map is not None:
        confidence = derive_confidence(
            [merged_change_map.mean_confidence],
            basis=f"merged change-map confidence across {len(change_map_dicts)} tiles",
        )
    else:
        source_confidences = [ev.confidence.value for ev in tile_evidence if ev.confidence.value is not None]
        confidence = derive_confidence(
            source_confidences,
            basis=f"mean of {len(source_confidences)} tile-level confidence values" if source_confidences else "",
        )

    return Evidence(
        task=task,
        model_used=model_used,
        modality_used=modality_used,
        detections=[Detection(**d) for d in kept_detections],
        change_map=merged_change_map,
        vqa_answer_raw=vqa_answer_raw,
        stats=merged_stats,
        confidence=Confidence(**confidence),
        warnings=warnings,
    )


def _mosaic_and_recount(raster_paths: list[str], image_id: str, total_image_px: int, threshold: float = 0.5):
    """
    Mosaics per-tile change-probability rasters into one full-extent raster
    via rasterio.merge (avoids the double-counting a plain sum of per-tile
    changed_area_px would have in overlap margins), then recomputes
    changed_area_px / changed_area_pct from the merged array directly using
    the given probability threshold. Returns None (caller falls back to the
    pure-aggregation figures) if the mosaic can't be produced.
    """
    try:
        import rasterio
        from rasterio.merge import merge as rio_merge

        from . import store

        sources = [rasterio.open(p) for p in raster_paths]
        try:
            mosaic_array, mosaic_transform = rio_merge(sources)
        finally:
            for s in sources:
                s.close()

        out_path = f"{store.stitched_dir_for(image_id)}/change_map_mosaic.tif"
        profile = {
            "driver": "GTiff",
            "height": mosaic_array.shape[1],
            "width": mosaic_array.shape[2],
            "count": 1,
            "dtype": str(mosaic_array.dtype),
            "crs": sources[0].crs if sources else None,
            "transform": mosaic_transform,
            "compress": "DEFLATE",
        }
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mosaic_array[0], 1)

        changed_px = int((mosaic_array[0] > threshold).sum())
        changed_pct = 100.0 * changed_px / total_image_px if total_image_px else 0.0
        return {
            "probability_raster_path": out_path,
            "changed_area_px": changed_px,
            "changed_area_pct": changed_pct,
        }
    except Exception:
        return None
