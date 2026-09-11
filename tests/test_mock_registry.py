"""
Proves Definition of Done #1 (Section 3.4): "schema-valid Evidence for
every mandatory capability." Also proves the mock's artificial-latency and
OOM-simulation knobs actually work, since Parts 2 and 5 build against
those directly.
"""
from __future__ import annotations

import time

import pytest

from backend.model_registry.mock_registry import MockInferenceEngine
from backend.shared.schemas import Evidence, Modality, TaskType

MANDATORY_TASKS = [
    TaskType.SINGLE_IMAGE_VQA,
    TaskType.CAPTIONING,
    TaskType.GROUNDING,
    TaskType.CHANGE_DETECTION,
    TaskType.CHANGE_VQA,
    TaskType.OPTICAL_SAR_FUSION,
]


def _tiles_for(task: TaskType, single_tile, before_after_tiles, optical_sar_tiles):
    if task in (TaskType.CHANGE_DETECTION, TaskType.CHANGE_VQA):
        return before_after_tiles
    if task == TaskType.OPTICAL_SAR_FUSION:
        return optical_sar_tiles
    return [single_tile]


@pytest.mark.parametrize("task", MANDATORY_TASKS)
def test_mock_returns_schema_valid_evidence_for_every_mandatory_task(task, single_tile, before_after_tiles, optical_sar_tiles):
    engine = MockInferenceEngine()
    tiles = _tiles_for(task, single_tile, before_after_tiles, optical_sar_tiles)

    evidence = engine.run_inference(task, tiles, query="what is here?")

    assert isinstance(evidence, Evidence)
    assert evidence.task == task
    assert evidence.model_used
    assert evidence.confidence.band in ("LOW", "MEDIUM", "HIGH")


def test_grounding_fixture_has_detections(single_tile):
    evidence = MockInferenceEngine().run_inference(TaskType.GROUNDING, [single_tile], query="find buildings")
    assert len(evidence.detections) > 0
    assert all(d.score >= 0.0 for d in evidence.detections)


def test_change_detection_fixture_has_change_map(before_after_tiles):
    evidence = MockInferenceEngine().run_inference(TaskType.CHANGE_DETECTION, before_after_tiles)
    assert evidence.change_map is not None
    assert evidence.change_map.changed_area_pct > 0


def test_vqa_fixtures_have_answer_text(single_tile):
    evidence = MockInferenceEngine().run_inference(TaskType.SINGLE_IMAGE_VQA, [single_tile], query="how many buildings?")
    assert evidence.vqa_answer_raw


def test_fusion_fixture_reports_both_modalities(optical_sar_tiles):
    evidence = MockInferenceEngine().run_inference(TaskType.OPTICAL_SAR_FUSION, optical_sar_tiles)
    assert set(evidence.modality_used) == {Modality.OPTICAL, Modality.SAR}


def test_list_available_models_covers_every_mandatory_task():
    models = MockInferenceEngine().list_available_models()
    covered = {t for entry in models for t in entry.tasks}
    assert set(MANDATORY_TASKS).issubset(covered)


def test_health_check_shape():
    health = MockInferenceEngine().health_check()
    assert health["status"] == "ok"
    assert isinstance(health["models_loaded"], list)


def test_artificial_latency_actually_delays(single_tile):
    engine = MockInferenceEngine(artificial_latency_s=0.05)
    start = time.monotonic()
    engine.run_inference(TaskType.CAPTIONING, [single_tile])
    assert time.monotonic() - start >= 0.045


def test_simulate_oom_for_raises_oom_style_error(single_tile):
    engine = MockInferenceEngine(simulate_oom_for={TaskType.CAPTIONING})
    with pytest.raises(RuntimeError, match="(?i)out of memory"):
        engine.run_inference(TaskType.CAPTIONING, [single_tile])


def test_run_inference_rejects_empty_tiles():
    with pytest.raises(ValueError):
        MockInferenceEngine().run_inference(TaskType.CAPTIONING, [])
