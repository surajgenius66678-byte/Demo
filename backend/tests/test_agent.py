"""
Run with `pytest` from backend/, after `pip install -r requirements.txt`.

These test Part 2's own logic against the mocked Part 3/4/5 functions, per
Section 3.2's own Mocking strategy — they don't (and can't yet) exercise
Parts 3/4/5's real behavior, and they don't drive api/main.py through real
HTTP (that file is a thin, directly-reviewable wrapper around what's tested
here; add httpx.AsyncClient-based tests against it once Parts 3/4/5 are
real and there's a reason to catch wiring mistakes at the HTTP layer too).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from shared.schemas import Confidence, Evidence, ImageMetadata, Modality, TaskType
from agent import mocks
from agent.intent import (
    HeuristicFallbackClassifier,
    IntentClassifier,
    NullLLMClassifier,
    _deterministic_classify,
    classify_intent,
)
from agent.planner import Part345Functions, Planner ,build_plan
from agent.queue import JobQueue
from agent.store import ImageStore, UploadSessionStore
from agent.trace import Trace


def make_image(modality=Modality.OPTICAL, image_id="img1", timestamp=None) -> ImageMetadata:
    return ImageMetadata(
        image_id=image_id, modality=modality, crs="EPSG:4326", bounds=[0, 0, 1, 1],
        width=100, height=100, band_count=4, dtype="uint16", resolution_m=10.0,
        timestamp=timestamp, cog_path=f"/data/{image_id}.tif", is_valid=True,
    )


def make_funcs(run_inference=None, check_coreg=None) -> Part345Functions:
    return Part345Functions(
        tile_image=mocks.mock_tile_image,
        check_coregistration=check_coreg or mocks.mock_check_coregistration,
        run_inference=run_inference or mocks.mock_run_inference,
        validate_and_respond=mocks.mock_validate_and_respond,
    )


# ==================================================================
# Intent classification
# ==================================================================

@pytest.mark.parametrize("query,expected", [
    ("find the buildings in this image", TaskType.GROUNDING),
    ("describe this image", TaskType.CAPTIONING),
    ("what is the population here?", TaskType.UNSUPPORTED),
])
def test_deterministic_classify_single_image(query, expected):
    task, confidence = _deterministic_classify(query, [make_image()])
    assert task == expected
    assert confidence >= 0.65


def test_deterministic_classify_ambiguous_single_image_is_low_confidence():
    # Genuine 3-way ambiguity (VQA vs captioning vs grounding) — should be
    # LOW confidence so it's handed to the LLM layer rather than guessed.
    task, confidence = _deterministic_classify("is this area mostly urban?", [make_image()])
    assert task == TaskType.SINGLE_IMAGE_VQA
    assert confidence < 0.65


def test_deterministic_classify_no_images_is_unsupported():
    task, _ = _deterministic_classify("anything", [])
    assert task == TaskType.UNSUPPORTED


def test_deterministic_classify_too_many_images_is_unsupported():
    images = [make_image(image_id=x) for x in ("a", "b", "c")]
    task, _ = _deterministic_classify("anything", images)
    assert task == TaskType.UNSUPPORTED


def test_deterministic_classify_change_vqa_vs_change_detection():
    pair = [make_image(Modality.OPTICAL, "a"), make_image(Modality.OPTICAL, "b")]
    task, _ = _deterministic_classify("what changed between these two images?", pair)
    assert task == TaskType.CHANGE_VQA
    task, _ = _deterministic_classify("detect changes", pair)
    assert task == TaskType.CHANGE_DETECTION


def test_deterministic_classify_fusion():
    pair = [make_image(Modality.OPTICAL, "a"), make_image(Modality.SAR, "b")]
    task, _ = _deterministic_classify("what is visible in this area?", pair)
    assert task == TaskType.OPTICAL_SAR_FUSION


async def test_llm_layer_not_invoked_when_deterministic_is_confident():
    class PoisonedLLM:
        async def classify(self, query, images):
            raise AssertionError("LLM layer should not be called when rules are confident")

    classifier = IntentClassifier(llm_classifier=PoisonedLLM())
    result = await classifier.classify_intent("find the buildings", [make_image()])
    assert result == TaskType.GROUNDING


async def test_neither_layer_confident_falls_back_to_unsupported():
    classifier = IntentClassifier(llm_classifier=NullLLMClassifier())
    result = await classifier.classify_intent("is this area mostly urban?", [make_image()])
    assert result == TaskType.UNSUPPORTED


async def test_llm_layer_resolves_ambiguous_case():
    classifier = IntentClassifier(llm_classifier=HeuristicFallbackClassifier())
    result = await classifier.classify_intent("is this area mostly urban?", [make_image()])
    assert result == TaskType.SINGLE_IMAGE_VQA


async def test_module_level_classify_intent_matches_section_3_2_signature():
    result = await classify_intent("find the road", [make_image()])
    assert result == TaskType.GROUNDING


# ==================================================================
# Planner sequencing
# ==================================================================

@pytest.mark.parametrize("task", [TaskType.SINGLE_IMAGE_VQA, TaskType.CAPTIONING, TaskType.GROUNDING])
async def test_planner_single_image_tasks_skip_coregistration(task):
    trace = Trace()
    resp = await Planner().execute(task, "q", [make_image()], trace, asyncio.Semaphore(2), make_funcs())
    assert resp.abstained is False
    assert [s.stage for s in trace.steps] == ["tiling", "inference", "evidence_aggregation","response_generation"]


async def test_planner_change_detection_when_aligned():
    pair = [make_image(Modality.OPTICAL, "a"), make_image(Modality.OPTICAL, "b")]
    trace = Trace()
    resp = await Planner().execute(
        TaskType.CHANGE_DETECTION, "what changed", pair, trace, asyncio.Semaphore(2), make_funcs()
    )
    assert resp.abstained is False
    assert [s.stage for s in trace.steps] == [
        "coregistration_check", "tiling", "inference", "evidence_aggregation","response_generation",
    ]


async def test_planner_passes_aggregated_evidence_to_part5():
    received = {}

    async def capture_validate(query, evidence):
        received["evidence"] = evidence

        return __import__("shared.schemas", fromlist=["FinalResponse"]).FinalResponse(
            answer_text="ok",
            confidence=evidence.confidence,
            evidence=evidence,
            abstained=False,
            abstain_reason=None,
        )

    async def multi_evidence_inference(
        task,
        tiles,
        model_hint=None,
        *,
        query=None,
        image_modalities=None,
        image_order=None,
    ):
        if task == TaskType.SINGLE_IMAGE_VQA:
            return Evidence(
                task=task,
                model_used="vqa-model",
                modality_used=[Modality.OPTICAL],
                vqa_answer_raw="There are buildings.",
                confidence=Confidence(
                    value=0.9,
                    band="HIGH",
                    basis="test",
                ),
            )

        return Evidence(
            task=task,
            model_used="grounding-model",
            modality_used=[Modality.OPTICAL],
            vqa_answer_raw=None,
            confidence=Confidence(
                value=0.8,
                band="HIGH",
                basis="test",
            ),
        )

    funcs = Part345Functions(
        tile_image=mocks.mock_tile_image,
        check_coregistration=mocks.mock_check_coregistration,
        run_inference=multi_evidence_inference,
        validate_and_respond=capture_validate,
    )

    trace = Trace()

    resp = await Planner().execute(
        TaskType.SINGLE_IMAGE_VQA,
        "where are the buildings?",
        [make_image()],
        trace,
        asyncio.Semaphore(2),
        funcs,
    )

    assert resp.abstained is False
    assert "evidence" in received

    aggregated = received["evidence"]

    # Evidence from the primary VQA specialist survives aggregation.
    assert "There are buildings." in aggregated.vqa_answer_raw

    # Both specialist models are represented.
    assert "vqa-model" in aggregated.model_used
    assert "grounding-model" in aggregated.model_used

    # Evidence from the grounding specialist survives aggregation.
    assert aggregated.modality_used == [Modality.OPTICAL]


async def test_planner_refuses_when_misaligned_without_calling_inference():
    pair = [make_image(Modality.OPTICAL, "misaligned-a"), make_image(Modality.OPTICAL, "b")]

    async def poisoned_run_inference(task, tiles, model_hint=None):
        raise AssertionError("run_inference must not be called on co-registration failure")

    trace = Trace()
    resp = await Planner().execute(
        TaskType.CHANGE_DETECTION, "what changed", pair, trace, asyncio.Semaphore(2),
        make_funcs(run_inference=poisoned_run_inference),
    )
    assert resp.abstained is True
    assert resp.abstain_reason
    assert [s.stage for s in trace.steps] == ["coregistration_check", "response_generation"]


async def test_planner_unsupported_short_circuits_with_no_downstream_calls():
    trace = Trace()
    resp = await Planner().execute(
        TaskType.UNSUPPORTED, "population?", [make_image()], trace, asyncio.Semaphore(2), make_funcs()
    )
    assert resp.abstained is True
    assert trace.steps == []


async def test_planner_fusion_is_also_coregistration_gated():
    pair = [make_image(Modality.OPTICAL, "a"), make_image(Modality.SAR, "b")]
    trace = Trace()
    resp = await Planner().execute(
        TaskType.OPTICAL_SAR_FUSION, "describe this area", pair, trace, asyncio.Semaphore(2), make_funcs()
    )
    assert resp.abstained is False
    assert [s.stage for s in trace.steps][0] == "coregistration_check"


# ==================================================================
# Job queue: bounded concurrency, HTTP decoupling, failure isolation
# ==================================================================

async def test_queue_job_keeps_running_with_no_one_polling_it():
    """Simulates 'closing the tab': nothing awaits or polls the job for a
    while after submit() returns, and it still completes on its own."""
    queue = JobQueue(funcs=make_funcs(), max_concurrent_inference=1)
    start = time.monotonic()
    job_id = queue.submit("describe this image", [make_image()])
    assert (time.monotonic() - start) * 1000 < 50  # submit() itself is near-instant

    await asyncio.sleep(0.4)  # no polling at all during this window
    assert queue.get(job_id).status == "done"


async def test_queue_bounds_concurrent_inference_calls():
    active = 0
    max_active = 0

    async def tracked_run_inference(
    task,
    tiles,
    model_hint=None,
    *,
    query=None,
    image_modalities=None,
    image_order=None,
):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.1)
        active -= 1
        return Evidence(
            task=task, model_used="mock", modality_used=[],
            vqa_answer_raw="ok", confidence=Confidence(value=0.9, band="HIGH", basis="test"),
        )

    queue = JobQueue(funcs=make_funcs(run_inference=tracked_run_inference), max_concurrent_inference=1)
    job_a = queue.submit("describe this image", [make_image(image_id="a")])
    job_b = queue.submit("describe this image", [make_image(image_id="b")])

    for _ in range(100):
        if queue.get(job_a).status in ("done", "failed") and queue.get(job_b).status in ("done", "failed"):
            break
        await asyncio.sleep(0.02)

    assert queue.get(job_a).status == "done"
    assert queue.get(job_b).status == "done"
    assert max_active == 1  # queued, never collided
    assert queue.get(job_a).result.abstained is False
    assert len(queue.get(job_a).trace.steps) > 0


async def test_queue_isolates_job_failures():
    async def failing_run_inference(
    task,
    tiles,
    model_hint=None,
    *,
    query=None,
    image_modalities=None,
    image_order=None,
):
        raise RuntimeError("simulated GPU OOM")

    queue = JobQueue(funcs=make_funcs(run_inference=failing_run_inference), max_concurrent_inference=1)
    job_id = queue.submit("describe this image", [make_image()])
    for _ in range(50):
        if queue.get(job_id).status in ("done", "failed"):
            break
        await asyncio.sleep(0.02)

    assert queue.get(job_id).status == "failed"
    assert "simulated GPU OOM" in queue.get(job_id).error

    # the queue itself keeps working after a job failure — a second submit
    # is still accepted and still runs to completion (of that same failing
    # mock), proving the failure didn't take the process/queue down
    job_id_2 = queue.submit("describe this image", [make_image(image_id="other")])
    for _ in range(50):
        if queue.get(job_id_2).status in ("done", "failed"):
            break
        await asyncio.sleep(0.02)
    assert queue.get(job_id_2).status == "failed"


# ==================================================================
# Stores
# ==================================================================

def test_image_store_roundtrip():
    store = ImageStore()
    metadata = make_image(image_id="xyz")
    store.put(metadata)
    assert store.get("xyz") is metadata
    assert store.get("nope") is None


def test_upload_session_reassembles_chunks_in_order(tmp_path):
    sessions = UploadSessionStore(tmp_path)
    original = b"0123456789" * 1000
    chunk_size = 4000
    chunks = [original[i:i + chunk_size] for i in range(0, len(original), chunk_size)]

    session = sessions.get_or_create("up1", len(chunks), Modality.OPTICAL, None)
    for i, chunk in enumerate(chunks):
        session.write_chunk(i, chunk)

    assert session.is_complete()
    assert session.assemble().read_bytes() == original


def test_upload_session_reports_missing_chunks_for_resumability(tmp_path):
    sessions = UploadSessionStore(tmp_path)
    session = sessions.get_or_create("up2", 3, Modality.SAR, None)
    session.write_chunk(0, b"a")
    session.write_chunk(2, b"c")
    assert session.received == {0, 2}
    assert not session.is_complete()

def test_build_plan_adds_grounding_for_spatial_query():
    plan = build_plan(
        TaskType.SINGLE_IMAGE_VQA,
        "where are the buildings?",
        [make_image()],
    )

    assert [task.task_type for task in plan.tasks] == [
        TaskType.SINGLE_IMAGE_VQA,
        TaskType.GROUNDING,
    ]


def test_build_plan_keeps_simple_vqa_single_task():
    plan = build_plan(
        TaskType.SINGLE_IMAGE_VQA,
        "what is visible in this image?",
        [make_image()],
    )

    assert [task.task_type for task in plan.tasks] == [
        TaskType.SINGLE_IMAGE_VQA,
    ]

def test_build_plan_change_detection():
    images = [
        make_image(image_id="before", modality=Modality.OPTICAL),
        make_image(image_id="after", modality=Modality.OPTICAL),
    ]

    plan = build_plan(
        TaskType.CHANGE_DETECTION,
        "what changed between these images?",
        images,
    )

    assert [task.task_type for task in plan.tasks] == [
        TaskType.CHANGE_DETECTION,
    ]


def test_build_plan_optical_sar_fusion():
    images = [
        make_image(image_id="optical", modality=Modality.OPTICAL),
        make_image(image_id="sar", modality=Modality.SAR),
    ]

    plan = build_plan(
        TaskType.OPTICAL_SAR_FUSION,
        "analyze these optical and SAR images together",
        images,
    )

    assert [task.task_type for task in plan.tasks] == [
        TaskType.OPTICAL_SAR_FUSION,
    ]

def test_build_plan_adds_change_detection_for_change_query():
    images = [
        make_image(
            image_id="before",
            modality=Modality.OPTICAL,
        ),
        make_image(
            image_id="after",
            modality=Modality.OPTICAL,
        ),
    ]

    plan = build_plan(
        TaskType.SINGLE_IMAGE_VQA,
        "what changed between these images?",
        images,
    )

    assert [task.task_type for task in plan.tasks] == [
        TaskType.SINGLE_IMAGE_VQA,
        TaskType.CHANGE_DETECTION,
    ]


def test_build_plan_adds_optical_sar_fusion_for_modalities():
    images = [
        make_image(
            image_id="optical",
            modality=Modality.OPTICAL,
        ),
        make_image(
            image_id="sar",
            modality=Modality.SAR,
        ),
    ]

    plan = build_plan(
        TaskType.SINGLE_IMAGE_VQA,
        "analyze these images together",
        images,
    )

    assert [task.task_type for task in plan.tasks] == [
        TaskType.SINGLE_IMAGE_VQA,
        TaskType.OPTICAL_SAR_FUSION,
    ]

def test_build_plan_grounding_depends_on_change_detection():
    images = [
        make_image(
            image_id="before",
            modality=Modality.OPTICAL,
        ),
        make_image(
            image_id="after",
            modality=Modality.OPTICAL,
        ),
    ]

    plan = build_plan(
        TaskType.SINGLE_IMAGE_VQA,
        "where are the buildings that changed?",
        images,
    )

    assert [task.task_type for task in plan.tasks] == [
        TaskType.SINGLE_IMAGE_VQA,
        TaskType.CHANGE_DETECTION,
        TaskType.GROUNDING,
    ]

    assert plan.tasks[1].task_id == "task_2"
    assert plan.tasks[2].depends_on == ["task_2"]