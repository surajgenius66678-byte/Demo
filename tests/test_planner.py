from __future__ import annotations

import asyncio

import pytest

from shared.schemas import (
    Confidence,
    CoregistrationResult,
    Evidence,
    ImageMetadata,
    Modality,
    FinalResponse,
    TaskType,
    Tile,
)

from agent.planner import (
    Part345Functions,
    Planner,
    build_plan,
)
from agent.trace import Trace


def make_image(
    image_id: str,
    modality: Modality = Modality.OPTICAL,
) -> ImageMetadata:
    return ImageMetadata(
        image_id=image_id,
        modality=modality,
        crs="EPSG:4326",
        bounds=[0.0, 0.0, 1.0, 1.0],
        width=1024,
        height=1024,
        band_count=3,
        dtype="uint8",
        resolution_m=1.0,
        timestamp=None,
        cog_path=f"/fake/{image_id}.tif",
        is_valid=True,
        validation_errors=[],
    )


def make_evidence(
    task: TaskType,
    model: str = "test-model",
    answer: str | None = None,
) -> Evidence:
    return Evidence(
        task=task,
        model_used=model,
        modality_used=[Modality.OPTICAL],
        vqa_answer_raw=answer,
        confidence=Confidence(
            value=0.9,
            band="HIGH",
            basis="test",
        ),
    )


@pytest.mark.asyncio
async def test_single_image_vqa_executes():
    image = make_image("img-1")
    trace = Trace()
    calls = []

    async def tile_image(image_id, task, max_input_px, overlap):
        calls.append(("tile", image_id, task))
        return [
            Tile(
                tile_id="tile-1",
                image_id=image_id,
                col_off=0,
                row_off=0,
                width=512,
                height=512,
                affine_transform=[1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                array_path="/fake/tile.npy",
            )
        ]

    async def check_coregistration(image_a, image_b):
        raise AssertionError("Coregistration should not run for single-image VQA")

    async def run_inference(task, tiles, **kwargs):
        calls.append(("inference", task, kwargs))
        return make_evidence(
            task,
            answer="There is a road network.",
        )

    async def validate_and_respond(query, evidence):
        return FinalResponse(
            answer_text=evidence.vqa_answer_raw or "",
            confidence=evidence.confidence,
            evidence=evidence,
            abstained=False,
        )

    funcs = Part345Functions(
        tile_image=tile_image,
        check_coregistration=check_coregistration,
        run_inference=run_inference,
        validate_and_respond=validate_and_respond,
    )

    response = await Planner().execute(
        task=TaskType.SINGLE_IMAGE_VQA,
        query="What is visible in this image?",
        images=[image],
        trace=trace,
        inference_gate=asyncio.Semaphore(1),
        funcs=funcs,
    )

    assert response.abstained is False
    assert response.answer_text == "There is a road network."
    assert any(call[0] == "inference" for call in calls)

    stages = [step.stage for step in trace.steps]
    assert "tiling" in stages
    assert "inference" in stages
    assert "evidence_aggregation" in stages
    assert "response_generation" in stages


def test_build_plan_adds_change_detection_for_temporal_query():
    images = [
        make_image("before"),
        make_image("after"),
    ]

    plan = build_plan(
        TaskType.SINGLE_IMAGE_VQA,
        "What changed between the before and after images?",
        images,
    )

    task_types = [task.task_type for task in plan.tasks]

    assert TaskType.SINGLE_IMAGE_VQA in task_types
    assert TaskType.CHANGE_DETECTION in task_types


def test_build_plan_adds_optical_sar_fusion():
    images = [
        make_image("optical", Modality.OPTICAL),
        make_image("sar", Modality.SAR),
    ]

    plan = build_plan(
        TaskType.SINGLE_IMAGE_VQA,
        "Analyze these images together.",
        images,
    )

    task_types = [task.task_type for task in plan.tasks]

    assert TaskType.OPTICAL_SAR_FUSION in task_types


@pytest.mark.asyncio
async def test_cross_image_task_refuses_wrong_image_count():
    image = make_image("only-one")
    trace = Trace()

    async def tile_image(*args):
        raise AssertionError("Tiling must not run after cross-image validation fails")

    async def check_coregistration(*args):
        raise AssertionError("Coregistration must not run with the wrong image count")

    async def run_inference(*args, **kwargs):
        raise AssertionError("Inference must not run after validation failure")

    async def validate_and_respond(*args):
        raise AssertionError("Part 5 must not handle planner-level refusal")

    funcs = Part345Functions(
        tile_image=tile_image,
        check_coregistration=check_coregistration,
        run_inference=run_inference,
        validate_and_respond=validate_and_respond,
    )

    response = await Planner().execute(
        task=TaskType.CHANGE_DETECTION,
        query="What changed?",
        images=[image],
        trace=trace,
        inference_gate=asyncio.Semaphore(1),
        funcs=funcs,
    )

    assert response.abstained is True
    assert response.abstain_reason is not None
    assert "requires exactly 2 images" in response.abstain_reason
    assert response.evidence.task == TaskType.CHANGE_DETECTION


@pytest.mark.asyncio
async def test_dependent_grounding_receives_upstream_evidence():
    images = [
        make_image("before"),
        make_image("after"),
    ]

    trace = Trace()
    inference_calls = []

    async def tile_image(image_id, task, max_input_px, overlap):
        return [
            Tile(
                tile_id=f"{image_id}-tile",
                image_id=image_id,
                col_off=0,
                row_off=0,
                width=512,
                height=512,
                affine_transform=[1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                array_path="/fake/tile.npy",
            )
        ]

    async def check_coregistration(image_a, image_b):
        return CoregistrationResult(
            aligned=True,
            offset_px=0.0,
            auto_corrected=False,
            reason=None,
        )

    async def run_inference(task, tiles, **kwargs):
        inference_calls.append(
            {
                "task": task,
                "upstream_evidence": kwargs.get("upstream_evidence", []),
            }
        )

        return make_evidence(
            task,
            answer=f"Evidence from {task.value}",
        )

    async def validate_and_respond(query, evidence):
        return FinalResponse(
            answer_text="verified",
            confidence=evidence.confidence,
            evidence=evidence,
            abstained=False,
        )

    funcs = Part345Functions(
        tile_image=tile_image,
        check_coregistration=check_coregistration,
        run_inference=run_inference,
        validate_and_respond=validate_and_respond,
    )

    response = await Planner().execute(
        task=TaskType.SINGLE_IMAGE_VQA,
        query="What changed and where did it change?",
        images=images,
        trace=trace,
        inference_gate=asyncio.Semaphore(1),
        funcs=funcs,
    )

    assert response.abstained is False

    grounding_calls = [
        call
        for call in inference_calls
        if call["task"] == TaskType.GROUNDING
    ]

    assert grounding_calls, "Expected planner to schedule grounding"

    grounding_upstream = grounding_calls[0]["upstream_evidence"]

    assert grounding_upstream
    assert grounding_upstream[0].task == TaskType.CHANGE_DETECTION
