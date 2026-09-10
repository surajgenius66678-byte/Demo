"""
Run with `pytest` from backend/, after `pip install -r requirements.txt`.

These tests cover Part 2's agent logic using mocked Part 3/4/5 functions.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from shared.schemas import (
    Confidence,
    Evidence,
    ImageMetadata,
    Modality,
    TaskType,
)

from agent import mocks
from agent.intent import (
    HeuristicFallbackClassifier,
    IntentClassifier,
    NullLLMClassifier,
    _deterministic_classify,
    classify_intent,
)
from agent.planner import (
    Part345Functions,
    Planner,
    build_plan,
)
from agent.queue import JobQueue
from agent.store import ImageStore, UploadSessionStore
from agent.trace import Trace


# ==================================================================
# Test helpers
# ==================================================================

def make_image(
    modality=Modality.OPTICAL,
    image_id="img1",
    timestamp=None,
) -> ImageMetadata:
    return ImageMetadata(
        image_id=image_id,
        modality=modality,
        crs="EPSG:4326",
        bounds=[0, 0, 1, 1],
        width=100,
        height=100,
        band_count=4,
        dtype="uint16",
        resolution_m=10.0,
        timestamp=timestamp,
        cog_path=f"/data/{image_id}.tif",
        is_valid=True,
    )


def make_funcs(
    run_inference=None,
    check_coreg=None,
) -> Part345Functions:
    return Part345Functions(
        tile_image=mocks.mock_tile_image,
        check_coregistration=(
            check_coreg or mocks.mock_check_coregistration
        ),
        run_inference=(
            run_inference or mocks.mock_run_inference
        ),
        validate_and_respond=mocks.mock_validate_and_respond,
    )


# ==================================================================
# Intent classification
# ==================================================================

@pytest.mark.parametrize(
    "query,expected",
    [
        (
            "find the buildings in this image",
            TaskType.GROUNDING,
        ),
        (
            "describe this image",
            TaskType.CAPTIONING,
        ),
        (
            "what is the population here?",
            TaskType.UNSUPPORTED,
        ),
    ],
)
def test_deterministic_classify_single_image(
    query,
    expected,
):
    task, confidence = _deterministic_classify(
        query,
        [make_image()],
    )

    assert task == expected
    assert confidence >= 0.65


def test_deterministic_classify_ambiguous_single_image_is_low_confidence():
    task, confidence = _deterministic_classify(
        "is this area mostly urban?",
        [make_image()],
    )

    assert task == TaskType.SINGLE_IMAGE_VQA
    assert confidence < 0.65


def test_deterministic_classify_no_images_is_unsupported():
    task, _ = _deterministic_classify(
        "anything",
        [],
    )

    assert task == TaskType.UNSUPPORTED


def test_deterministic_classify_too_many_images_is_unsupported():
    images = [
        make_image(image_id=image_id)
        for image_id in ("a", "b", "c")
    ]

    task, _ = _deterministic_classify(
        "anything",
        images,
    )

    assert task == TaskType.UNSUPPORTED


def test_deterministic_classify_change_vqa_vs_change_detection():
    pair = [
        make_image(
            Modality.OPTICAL,
            "a",
        ),
        make_image(
            Modality.OPTICAL,
            "b",
        ),
    ]

    task, _ = _deterministic_classify(
        "what changed between these two images?",
        pair,
    )

    assert task == TaskType.CHANGE_VQA

    task, _ = _deterministic_classify(
        "detect changes",
        pair,
    )

    assert task == TaskType.CHANGE_DETECTION


def test_deterministic_classify_fusion():
    pair = [
        make_image(
            Modality.OPTICAL,
            "a",
        ),
        make_image(
            Modality.SAR,
            "b",
        ),
    ]

    task, _ = _deterministic_classify(
        "what is visible in this area?",
        pair,
    )

    assert task == TaskType.OPTICAL_SAR_FUSION


async def test_llm_layer_not_invoked_when_deterministic_is_confident():
    class PoisonedLLM:
        async def classify(self, query, images):
            raise AssertionError(
                "LLM layer should not be called "
                "when rules are confident"
            )

    classifier = IntentClassifier(
        llm_classifier=PoisonedLLM(),
    )

    result = await classifier.classify_intent(
        "find the buildings",
        [make_image()],
    )

    assert result == TaskType.GROUNDING


async def test_neither_layer_confident_falls_back_to_unsupported():
    classifier = IntentClassifier(
        llm_classifier=NullLLMClassifier(),
    )

    result = await classifier.classify_intent(
        "is this area mostly urban?",
        [make_image()],
    )

    assert result == TaskType.UNSUPPORTED


async def test_llm_layer_resolves_ambiguous_case():
    classifier = IntentClassifier(
        llm_classifier=HeuristicFallbackClassifier(),
    )

    result = await classifier.classify_intent(
        "is this area mostly urban?",
        [make_image()],
    )

    assert result == TaskType.SINGLE_IMAGE_VQA


async def test_module_level_classify_intent_matches_section_3_2_signature():
    result = await classify_intent(
        "find the road",
        [make_image()],
    )

    assert result == TaskType.GROUNDING


# ==================================================================
# Planner sequencing
# ==================================================================

@pytest.mark.parametrize(
    "task",
    [
        TaskType.SINGLE_IMAGE_VQA,
        TaskType.CAPTIONING,
        TaskType.GROUNDING,
    ],
)
async def test_planner_single_image_tasks_skip_coregistration(
    task,
):
    trace = Trace()

    response = await Planner().execute(
        task,
        "q",
        [make_image()],
        trace,
        asyncio.Semaphore(2),
        make_funcs(),
    )

    assert response.abstained is False

    assert [step.stage for step in trace.steps] == [
        "tiling",
        "inference",
        "evidence_aggregation",
        "response_generation",
    ]


async def test_planner_change_detection_when_aligned():
    pair = [
        make_image(
            Modality.OPTICAL,
            "a",
        ),
        make_image(
            Modality.OPTICAL,
            "b",
        ),
    ]

    trace = Trace()

    response = await Planner().execute(
        TaskType.CHANGE_DETECTION,
        "what changed",
        pair,
        trace,
        asyncio.Semaphore(2),
        make_funcs(),
    )

    assert response.abstained is False

    assert [step.stage for step in trace.steps] == [
        "coregistration_check",
        "tiling",
        "inference",
        "evidence_aggregation",
        "response_generation",
    ]


async def test_planner_passes_aggregated_evidence_to_part5():
    received = {}

    async def capture_validate(query, evidence):
        received["evidence"] = evidence

        FinalResponse = __import__(
            "shared.schemas",
            fromlist=["FinalResponse"],
        ).FinalResponse

        return FinalResponse(
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
        upstream_evidence=None,
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

    response = await Planner().execute(
        TaskType.SINGLE_IMAGE_VQA,
        "where are the buildings?",
        [make_image()],
        trace,
        asyncio.Semaphore(2),
        funcs,
    )

    assert response.abstained is False
    assert "evidence" in received

    aggregated = received["evidence"]

    assert "There are buildings." in aggregated.vqa_answer_raw

    assert "vqa-model" in aggregated.model_used
    assert "grounding-model" in aggregated.model_used

    assert aggregated.modality_used == [
        Modality.OPTICAL
    ]


async def test_planner_refuses_when_misaligned_without_calling_inference():
    pair = [
        make_image(
            Modality.OPTICAL,
            "misaligned-a",
        ),
        make_image(
            Modality.OPTICAL,
            "b",
        ),
    ]

    async def poisoned_run_inference(
        task,
        tiles,
        model_hint=None,
        *,
        query=None,
        image_modalities=None,
        image_order=None,
        upstream_evidence=None,
    ):
        raise AssertionError(
            "run_inference must not be called "
            "on co-registration failure"
        )

    trace = Trace()

    response = await Planner().execute(
        TaskType.CHANGE_DETECTION,
        "what changed",
        pair,
        trace,
        asyncio.Semaphore(2),
        make_funcs(
            run_inference=poisoned_run_inference,
        ),
    )

    assert response.abstained is True
    assert response.abstain_reason

    assert [step.stage for step in trace.steps] == [
        "coregistration_check",
        "response_generation",
    ]


async def test_planner_unsupported_short_circuits_with_no_downstream_calls():
    trace = Trace()

    response = await Planner().execute(
        TaskType.UNSUPPORTED,
        "population?",
        [make_image()],
        trace,
        asyncio.Semaphore(2),
        make_funcs(),
    )

    assert response.abstained is True
    assert trace.steps == []


async def test_planner_fusion_is_also_coregistration_gated():
    pair = [
        make_image(
            Modality.OPTICAL,
            "a",
        ),
        make_image(
            Modality.SAR,
            "b",
        ),
    ]

    trace = Trace()

    response = await Planner().execute(
        TaskType.OPTICAL_SAR_FUSION,
        "describe this area",
        pair,
        trace,
        asyncio.Semaphore(2),
        make_funcs(),
    )

    assert response.abstained is False
    assert trace.steps[0].stage == "coregistration_check"


# ==================================================================
# Mandatory SIH planner scenarios
# ==================================================================

def test_plan_single_image_vqa():
    plan = build_plan(
        TaskType.SINGLE_IMAGE_VQA,
        "What is visible in this image?",
        [
            make_image(
                Modality.OPTICAL,
                "img1",
            )
        ],
    )

    assert [task.task_type for task in plan.tasks] == [
        TaskType.SINGLE_IMAGE_VQA,
    ]


def test_plan_single_image_grounding():
    plan = build_plan(
        TaskType.SINGLE_IMAGE_VQA,
        "Where are the buildings?",
        [
            make_image(
                Modality.OPTICAL,
                "img1",
            )
        ],
    )

    assert [task.task_type for task in plan.tasks] == [
        TaskType.SINGLE_IMAGE_VQA,
        TaskType.GROUNDING,
    ]


def test_plan_bitemporal_change():
    plan = build_plan(
        TaskType.SINGLE_IMAGE_VQA,
        "What changed between these images?",
        [
            make_image(
                Modality.OPTICAL,
                "before",
            ),
            make_image(
                Modality.OPTICAL,
                "after",
            ),
        ],
    )

    assert [task.task_type for task in plan.tasks] == [
        TaskType.SINGLE_IMAGE_VQA,
        TaskType.CHANGE_DETECTION,
    ]


def test_plan_optical_sar_fusion():
    plan = build_plan(
        TaskType.SINGLE_IMAGE_VQA,
        "Analyze these images together.",
        [
            make_image(
                Modality.OPTICAL,
                "optical",
            ),
            make_image(
                Modality.SAR,
                "sar",
            ),
        ],
    )

    assert [task.task_type for task in plan.tasks] == [
        TaskType.SINGLE_IMAGE_VQA,
        TaskType.OPTICAL_SAR_FUSION,
    ]


def test_plan_grounding_after_change():
    plan = build_plan(
        TaskType.SINGLE_IMAGE_VQA,
        "Where are the buildings that changed?",
        [
            make_image(
                Modality.OPTICAL,
                "before",
            ),
            make_image(
                Modality.OPTICAL,
                "after",
            ),
        ],
    )

    assert [task.task_type for task in plan.tasks] == [
        TaskType.SINGLE_IMAGE_VQA,
        TaskType.CHANGE_DETECTION,
        TaskType.GROUNDING,
    ]

    change_task = next(
        task
        for task in plan.tasks
        if task.task_type == TaskType.CHANGE_DETECTION
    )

    grounding_task = next(
        task
        for task in plan.tasks
        if task.task_type == TaskType.GROUNDING
    )

    assert grounding_task.depends_on == [
        change_task.task_id
    ]


# ==================================================================
# Grounding upstream evidence
# ==================================================================

def test_grounding_adapter_uses_upstream_evidence():
    from backend.model_registry.adapters.grounding_adapter import (
        GroundingAdapter,
    )

    class FakeModel:
        def __init__(self):
            self.received_queries = []

        def ground(self, image_path, query):
            self.received_queries.append(query)
            return []

    upstream = Evidence(
        task=TaskType.CHANGE_DETECTION,
        model_used="change-model",
        modality_used=[Modality.OPTICAL],
        confidence=Confidence(
            value=0.9,
            band="HIGH",
            basis="test",
        ),
        warnings=[],
    )

    adapter = GroundingAdapter(
        query="locate the buildings that changed",
        upstream_evidence=[upstream],
    )

    fake_model = FakeModel()

    adapter.infer(
        fake_model,
        [
            {
                "tile": None,
                "image_path": "fake-image.tif",
            }
        ],
    )

    assert len(fake_model.received_queries) == 1

    received_query = fake_model.received_queries[0]

    assert (
        "locate the buildings that changed"
        in received_query
    )

    assert (
        "Previous task: CHANGE_DETECTION"
        in received_query
    )


# ==================================================================
# Job queue
# ==================================================================

async def test_queue_job_keeps_running_with_no_one_polling_it():
    """
    Simulates closing the tab: nothing awaits or polls the job
    for a while after submit(), and it still completes.
    """

    queue = JobQueue(
        funcs=make_funcs(),
        max_concurrent_inference=1,
    )

    start = time.monotonic()

    job_id = queue.submit(
        "describe this image",
        [make_image()],
    )

    assert (
        (time.monotonic() - start) * 1000
        < 50
    )

    await asyncio.sleep(0.4)

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
        upstream_evidence=None,
    ):
        nonlocal active, max_active

        active += 1
        max_active = max(
            max_active,
            active,
        )

        await asyncio.sleep(0.1)

        active -= 1

        return Evidence(
            task=task,
            model_used="mock",
            modality_used=[],
            vqa_answer_raw="ok",
            confidence=Confidence(
                value=0.9,
                band="HIGH",
                basis="test",
            ),
        )

    queue = JobQueue(
        funcs=make_funcs(
            run_inference=tracked_run_inference,
        ),
        max_concurrent_inference=1,
    )

    job_a = queue.submit(
        "describe this image",
        [make_image(image_id="a")],
    )

    job_b = queue.submit(
        "describe this image",
        [make_image(image_id="b")],
    )

    for _ in range(100):
        status_a = queue.get(job_a).status
        status_b = queue.get(job_b).status

        if (
            status_a in ("done", "failed")
            and status_b in ("done", "failed")
        ):
            break

        await asyncio.sleep(0.02)

    assert queue.get(job_a).status == "done"
    assert queue.get(job_b).status == "done"

    assert max_active == 1

    assert queue.get(job_a).result.abstained is False

    assert len(
        queue.get(job_a).trace.steps
    ) > 0


async def test_queue_isolates_job_failures():
    async def failing_run_inference(
        task,
        tiles,
        model_hint=None,
        *,
        query=None,
        image_modalities=None,
        image_order=None,
        upstream_evidence=None,
    ):
        raise RuntimeError(
            "simulated GPU OOM"
        )

    queue = JobQueue(
        funcs=make_funcs(
            run_inference=failing_run_inference,
        ),
        max_concurrent_inference=1,
    )

    job_id = queue.submit(
        "describe this image",
        [make_image()],
    )

    for _ in range(50):
        if queue.get(job_id).status in (
            "done",
            "failed",
        ):
            break

        await asyncio.sleep(0.02)

    assert queue.get(job_id).status == "failed"

    assert (
        "simulated GPU OOM"
        in queue.get(job_id).error
    )

    job_id_2 = queue.submit(
        "describe this image",
        [make_image(image_id="other")],
    )

    for _ in range(50):
        if queue.get(job_id_2).status in (
            "done",
            "failed",
        ):
            break

        await asyncio.sleep(0.02)

    assert queue.get(job_id_2).status == "failed"


# ==================================================================
# Stores
# ==================================================================

def test_image_store_roundtrip():
    store = ImageStore()

    metadata = make_image(
        image_id="xyz"
    )

    store.put(metadata)

    assert store.get("xyz") is metadata
    assert store.get("nope") is None


def test_upload_session_reassembles_chunks_in_order(tmp_path):
    sessions = UploadSessionStore(tmp_path)

    original = b"0123456789" * 1000

    chunk_size = 4000

    chunks = [
        original[i:i + chunk_size]
        for i in range(
            0,
            len(original),
            chunk_size,
        )
    ]

    session = sessions.get_or_create(
        "up1",
        len(chunks),
        Modality.OPTICAL,
        None,
    )

    for index, chunk in enumerate(chunks):
        session.write_chunk(
            index,
            chunk,
        )

    assert session.is_complete()

    assert (
        session.assemble().read_bytes()
        == original
    )


def test_upload_session_reports_missing_chunks_for_resumability(
    tmp_path,
):
    sessions = UploadSessionStore(tmp_path)

    session = sessions.get_or_create(
        "up2",
        3,
        Modality.SAR,
        None,
    )

    session.write_chunk(
        0,
        b"a",
    )

    session.write_chunk(
        2,
        b"c",
    )

    assert session.received == {
        0,
        2,
    }

    assert not session.is_complete()


# ==================================================================
# Basic plan tests
# ==================================================================

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
        TaskType.CHANGE_DETECTION,
        "what changed between these images?",
        images,
    )

    assert [task.task_type for task in plan.tasks] == [
        TaskType.CHANGE_DETECTION,
    ]


def test_build_plan_optical_sar_fusion():
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

    assert plan.tasks[2].depends_on == [
        "task_2"
    ]

@pytest.mark.asyncio
async def test_change_detection_requires_exactly_two_images():
    planner = Planner()
    trace = Trace()
    gate = asyncio.Semaphore(1)

    called = False

    async def run_inference(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError(
            "inference must not run for invalid image count"
        )

    funcs = make_funcs(
        run_inference=run_inference,
    )

    images = [
        make_image(image_id="img1"),
    ]

    response = await planner.execute(
        TaskType.CHANGE_DETECTION,
        "detect changes",
        images,
        trace,
        gate,
        funcs,
    )

    assert response.abstained is True
    assert called is False


@pytest.mark.asyncio
async def test_optical_sar_fusion_requires_exactly_two_images():
    planner = Planner()
    trace = Trace()
    gate = asyncio.Semaphore(1)

    called = False

    async def run_inference(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError(
            "inference must not run for invalid image count"
        )

    funcs = make_funcs(
        run_inference=run_inference,
    )

    images = [
        make_image(
            image_id="optical",
            modality=Modality.OPTICAL,
        ),
    ]

    response = await planner.execute(
        TaskType.OPTICAL_SAR_FUSION,
        "analyze optical and SAR data",
        images,
        trace,
        gate,
        funcs,
    )

    assert response.abstained is True
    assert called is False


@pytest.mark.asyncio
async def test_change_vqa_requires_exactly_two_images():
    planner = Planner()
    trace = Trace()
    gate = asyncio.Semaphore(1)

    called = False

    async def run_inference(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError(
            "inference must not run for invalid image count"
        )

    funcs = make_funcs(
        run_inference=run_inference,
    )

    images = [
        make_image(image_id="img1"),
        make_image(image_id="img2"),
        make_image(image_id="img3"),
    ]

    response = await planner.execute(
        TaskType.CHANGE_VQA,
        "what changed?",
        images,
        trace,
        gate,
        funcs,
    )

    assert response.abstained is True
    assert called is False