"""
Stub implementations of every function Part 2 depends on but does not own:
validate_and_prepare / tile_image / check_coregistration / run_inference /
validate_and_respond (Section 3.2's exact list), plus list_available_models
and health_check, which Part 4's own interface (Section 3.4) defines and
which Part 2's GET /api/models and GET /api/health endpoints need even
though the abbreviated call list in 3.2 doesn't repeat them.

Per Section 3.2's Mocking strategy: "implement ... as stubs returning
hardcoded but schema-valid objects. Build and test the whole planner/queue/
API against these before Parts 3/4/5 exist." That is the only job of this
file — it is not a preview of what Part 3/4/5's real logic should look like.

Integration (Section 5, step 3): swap these imports in api/main.py and
agent/planner.py for the real Part 3/4/5 modules. Every function here is
`async def` so the swap is a straight signature match against an async
real implementation; if a real implementation ends up sync (e.g. a
rasterio/PyTorch call), wrap it with `asyncio.to_thread(...)` at the call
site rather than changing these signatures.

`mock_check_coregistration` treats an image_id containing the substring
"misaligned" as deliberately unaligned, so the co-registration-refusal path
(Section 3.3 hardening / Section 3.2's "Part 2 is responsible for refusing
the downstream task on that result") is exercisable in tests without a real
geospatial pipeline.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Optional

from shared.schemas import (
    ChangeMap,
    Confidence,
    CoregistrationResult,
    Detection,
    Evidence,
    FinalResponse,
    ImageMetadata,
    Modality,
    ModelRegistryEntry,
    TaskType,
    Tile,
)

# Simulated latency, in seconds. run_inference's is by far the largest since
# it's standing in for the GPU-bound call the bounded-concurrency queue exists
# to protect — real tests use this to prove two concurrent jobs actually queue.
_LATENCY = {"validate": 0.05, "tile": 0.02, "coreg": 0.02, "infer": 0.15, "respond": 0.02}


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


async def mock_validate_and_prepare(
    file_path: str, declared_modality: Modality, declared_timestamp: Optional[str]
) -> ImageMetadata:
    await asyncio.sleep(_LATENCY["validate"])
    data = Path(file_path).read_bytes()
    image_id = _hash_bytes(data)
    return ImageMetadata(
        image_id=image_id,
        modality=declared_modality,
        crs="EPSG:4326",
        bounds=[0.0, 0.0, 1.0, 1.0],
        width=4096,
        height=4096,
        band_count=1 if declared_modality == Modality.SAR else 4,
        dtype="uint16",
        resolution_m=10.0,
        timestamp=declared_timestamp,
        cog_path=f"/data/cogs/{image_id}.tif",
        is_valid=True,
        validation_errors=[],
    )


async def mock_tile_image(
    image_id: str, task: TaskType, tile_size: int = 1024, overlap_pct: float = 0.15
) -> list[Tile]:
    await asyncio.sleep(_LATENCY["tile"])
    step = int(tile_size * (1 - overlap_pct))
    return [
        Tile(
            tile_id=f"{image_id}-t{i}",
            image_id=image_id,
            col_off=i * step,
            row_off=0,
            width=tile_size,
            height=tile_size,
            affine_transform=[10.0, 0.0, float(i * step * 10), 0.0, -10.0, 0.0],
            array_path=f"/data/tiles/{image_id}-t{i}.npy",
        )
        for i in range(2)
    ]


async def mock_check_coregistration(image_a_id: str, image_b_id: str) -> CoregistrationResult:
    await asyncio.sleep(_LATENCY["coreg"])
    if "misaligned" in image_a_id or "misaligned" in image_b_id:
        return CoregistrationResult(
            aligned=False, offset_px=37.5, auto_corrected=False,
            reason="offset exceeds threshold (12px)",
        )
    return CoregistrationResult(aligned=True, offset_px=0.8, auto_corrected=False, reason=None)


async def mock_run_inference(
    task: TaskType,
    tiles: list[Tile],
    model_hint: Optional[str] = None,
    *,
    query: Optional[str] = None,
    image_modalities: Optional[dict[str, Modality]] = None,
    image_order: Optional[list[str]] = None,
    upstream_evidence: Optional[list[Evidence]] = None,
) -> Evidence:
    await asyncio.sleep(_LATENCY["infer"])
    image_ids = {t.image_id for t in tiles}
    modality_used = [Modality.OPTICAL]  # mock has no real pixels to inspect; a fixed stand-in

    if task == TaskType.GROUNDING:
        return Evidence(
            task=task, model_used="mock-grounding-v0", modality_used=modality_used,
            detections=[Detection(label="object", box_px=[100.0, 100.0, 220.0, 240.0],
                                   mask_rle=None, score=0.82)],
            confidence=Confidence(value=0.82, band="MEDIUM", basis="mock detector score"),
        )
    if task in (TaskType.CHANGE_DETECTION, TaskType.CHANGE_VQA):
        return Evidence(
            task=task, model_used="mock-change-v0", modality_used=modality_used,
            change_map=ChangeMap(probability_raster_path="/data/changemaps/mock.tif",
                                  changed_area_px=15000, changed_area_pct=4.3, mean_confidence=0.77),
            vqa_answer_raw="an area of new construction in the eastern part of the scene" if task == TaskType.CHANGE_VQA else None,
            stats={"changed_area_pct": 4.3},
            confidence=Confidence(value=0.77, band="MEDIUM", basis="mock change-map mean confidence"),
        )
    if task == TaskType.OPTICAL_SAR_FUSION:
        return Evidence(
            task=task, model_used="mock-fusion-v0", modality_used=[Modality.OPTICAL, Modality.SAR],
            vqa_answer_raw="a flooded field visible in SAR but obscured by cloud in the optical image",
            stats={"fused_tiles": float(len(tiles))},
            confidence=Confidence(value=0.7, band="MEDIUM", basis="mock fusion agreement score"),
        )
    # SINGLE_IMAGE_VQA / CAPTIONING
    return Evidence(
        task=task, model_used="mock-vlm-v0", modality_used=modality_used,
        vqa_answer_raw="a mostly rural scene with a river running through cultivated fields",
        stats={"tiles_used": float(len(tiles)), "images_used": float(len(image_ids))},
        confidence=Confidence(value=0.68, band="MEDIUM", basis="mock VLM logprob-derived score"),
    )


async def mock_validate_and_respond(query: str, evidence: Evidence) -> FinalResponse:
    await asyncio.sleep(_LATENCY["respond"])

    if any("co-registration" in w for w in evidence.warnings):
        return FinalResponse(
            answer_text="I can't reliably answer this — the two images aren't spatially aligned "
                        "closely enough to compare pixel-for-pixel.",
            confidence=Confidence(value=None, band="LOW", basis="co-registration check failed"),
            evidence=evidence, abstained=True, abstain_reason="co-registration failed",
        )

    if evidence.change_map is not None:
        text = (f"About {evidence.change_map.changed_area_pct:.1f}% of the scene changed. "
                f"{evidence.vqa_answer_raw or ''}").strip()
    elif evidence.detections:
        text = f"Found {len(evidence.detections)} matching object(s) in the image."
    elif evidence.vqa_answer_raw:
        text = evidence.vqa_answer_raw
    else:
        text = "No answer could be generated from the available evidence."

    return FinalResponse(
        answer_text=text, confidence=evidence.confidence, evidence=evidence,
        abstained=False, abstain_reason=None,
    )


async def mock_list_available_models() -> list[ModelRegistryEntry]:
    return [
        ModelRegistryEntry(
            name="mock-vlm", version="0.1.0",
            tasks=[TaskType.SINGLE_IMAGE_VQA, TaskType.CAPTIONING, TaskType.GROUNDING,
                   TaskType.CHANGE_VQA],
            modalities=[Modality.OPTICAL, Modality.SAR],
            checkpoint_path="/models/mock-vlm", quantization="8bit",
            requires_coregistration=False, max_input_px=1024,
        ),
        ModelRegistryEntry(
            name="mock-change-detector", version="0.1.0",
            tasks=[TaskType.CHANGE_DETECTION], modalities=[Modality.OPTICAL, Modality.SAR],
            checkpoint_path="/models/mock-change-detector", quantization="none",
            requires_coregistration=True, max_input_px=1024,
        ),
    ]


async def mock_health_check() -> dict:
    return {"status": "ok", "models_loaded": ["mock-vlm", "mock-change-detector"]}
