"""
Part 4 — mock registry (Section 3.4's mandated mocking strategy).

Returns hardcoded-but-schema-valid Evidence per TaskType so Parts 2 and 5
can be built and demoed end-to-end with zero GPU and zero real checkpoints.
Flip `configure_engine(use_mock=False)` once real models are ready — Part
2's call sites never change, only inference.py's own bookkeeping does.
"""
from __future__ import annotations

import time
from typing import Optional

from backend.shared.schemas import (
    ChangeMap, Confidence, Detection, Evidence, Modality,
    ModelRegistryEntry, TaskType, Tile,
)

_MOCK_REGISTRY: list[ModelRegistryEntry] = [
    ModelRegistryEntry(
        name="mock-vlm", version="0.0.0-mock",
        tasks=[TaskType.SINGLE_IMAGE_VQA, TaskType.CAPTIONING],
        modalities=[Modality.OPTICAL, Modality.SAR],
        checkpoint_path="(mock)", quantization="none",
        requires_coregistration=False, max_input_px=1024,
    ),
    ModelRegistryEntry(
        name="mock-grounding", version="0.0.0-mock",
        tasks=[TaskType.GROUNDING],
        modalities=[Modality.OPTICAL, Modality.SAR],
        checkpoint_path="(mock)", quantization="none",
        requires_coregistration=False, max_input_px=1024,
    ),
    ModelRegistryEntry(
        name="mock-change", version="0.0.0-mock",
        tasks=[TaskType.CHANGE_DETECTION, TaskType.CHANGE_VQA],
        modalities=[Modality.OPTICAL],
        checkpoint_path="(mock)", quantization="none",
        requires_coregistration=True, max_input_px=1024,
    ),
    ModelRegistryEntry(
        name="mock-fusion", version="0.0.0-mock",
        tasks=[TaskType.OPTICAL_SAR_FUSION],
        modalities=[Modality.OPTICAL, Modality.SAR],
        checkpoint_path="(mock)", quantization="none",
        requires_coregistration=True, max_input_px=1024,
    ),
]

_MODEL_FOR_TASK: dict[TaskType, str] = {
    TaskType.SINGLE_IMAGE_VQA: "mock-vlm",
    TaskType.CAPTIONING: "mock-vlm",
    TaskType.GROUNDING: "mock-grounding",
    TaskType.CHANGE_DETECTION: "mock-change",
    TaskType.CHANGE_VQA: "mock-change",
    TaskType.OPTICAL_SAR_FUSION: "mock-fusion",
}


class MockInferenceEngine:
    """Drop-in stand-in for InferenceEngine — same 3-method public interface."""

    def __init__(self, artificial_latency_s: float = 0.0, simulate_oom_for: Optional[set] = None):
        self.artificial_latency_s = artificial_latency_s
        self.simulate_oom_for = simulate_oom_for or set()

    def run_inference(self, task: TaskType, tiles: list[Tile], model_hint: Optional[str] = None, **kwargs) -> Evidence:
        if self.artificial_latency_s:
            time.sleep(self.artificial_latency_s)
        if task in self.simulate_oom_for:
            raise RuntimeError("CUDA out of memory (simulated by MockInferenceEngine for downstream testing)")
        if task not in _MODEL_FOR_TASK:
            raise ValueError(f"Mock registry has no fixture for task={task!r}")
        if not tiles:
            raise ValueError("run_inference requires at least one tile.")
        model_name = model_hint or _MODEL_FOR_TASK[task]
        return _fixture_evidence(task, model_name, tiles, kwargs.get("query"))

    def list_available_models(self) -> list[ModelRegistryEntry]:
        return list(_MOCK_REGISTRY)

    def health_check(self) -> dict:
        return {
            "status": "ok",
            "models_loaded": [e.name for e in _MOCK_REGISTRY],
            "vram_used_mb": 0.0,
            "vram_budget_mb": 0.0,
        }


def build_mock_engine(**kwargs) -> MockInferenceEngine:
    return MockInferenceEngine(**kwargs)


def _fixture_evidence(task: TaskType, model_name: str, tiles: list[Tile], query: Optional[str]) -> Evidence:
    tile = tiles[0]

    if task == TaskType.SINGLE_IMAGE_VQA:
        return Evidence(
            task=task, model_used=model_name, modality_used=[Modality.OPTICAL],
            vqa_answer_raw=f"[mock] Answer to '{query or 'the query'}': the scene shows mixed built-up and vegetated land.",
            confidence=Confidence(value=0.71, band="MEDIUM", basis="mock VLM decoder score"),
        )

    if task == TaskType.CAPTIONING:
        return Evidence(
            task=task, model_used=model_name, modality_used=[Modality.OPTICAL],
            vqa_answer_raw="[mock] A satellite view of a river cutting through farmland, with a small settlement to the northeast.",
            confidence=Confidence(value=0.80, band="HIGH", basis="mock VLM decoder score"),
        )

    if task == TaskType.GROUNDING:
        detections = [
            Detection(label="building", box_px=[tile.col_off + 12, tile.row_off + 8, tile.col_off + 96, tile.row_off + 80], mask_rle=None, score=0.88),
            Detection(label="building", box_px=[tile.col_off + 150, tile.row_off + 40, tile.col_off + 210, tile.row_off + 110], mask_rle=None, score=0.63),
        ]
        return Evidence(
            task=task, model_used=model_name, modality_used=[Modality.OPTICAL], detections=detections,
            confidence=Confidence(value=0.755, band="HIGH", basis="mean of 2 mock detection scores"),
        )

    if task in (TaskType.CHANGE_DETECTION, TaskType.CHANGE_VQA):
        change_map = ChangeMap(probability_raster_path=None, changed_area_px=48211, changed_area_pct=6.4, mean_confidence=0.69)
        answer = f"[mock] Change answer for '{query}': roughly 6.4% of the scene changed, concentrated in the northeast quadrant." if query else None
        return Evidence(
            task=task, model_used=model_name, modality_used=[Modality.OPTICAL], change_map=change_map,
            vqa_answer_raw=answer,
            confidence=Confidence(value=0.69, band="MEDIUM", basis="mock change-model confidence"),
        )

    if task == TaskType.OPTICAL_SAR_FUSION:
        detections = [Detection(label="flooded_area", box_px=[tile.col_off + 5, tile.row_off + 5, tile.col_off + 300, tile.row_off + 220], mask_rle=None, score=0.74)]
        return Evidence(
            task=task, model_used=model_name, modality_used=[Modality.OPTICAL, Modality.SAR], detections=detections,
            vqa_answer_raw="[mock] SAR confirms standing water under cloud cover the optical image alone couldn't resolve.",
            confidence=Confidence(value=0.74, band="MEDIUM", basis="mock fused-detection score"),
        )

    raise AssertionError(f"unreachable: task {task!r} not handled")
