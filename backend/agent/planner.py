"""
Planner.execute() sequences Part 3 -> Part 4 -> Part 5 calls for the
currently planned tasks.

The planner first builds an explicit AgentPlan. The execution layer then
walks through every PlannedTask, executes its specialist, stores the
resulting Evidence, and finally passes the primary evidence to Part 5.

This is the transition point from the original single-task planner toward
the final multi-specialist evidence aggregation architecture.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable

from shared.schemas import (
    Confidence,
    CoregistrationResult,
    Evidence,
    FinalResponse,
    ImageMetadata,
    TaskType,
    Tile,
)
from agent.trace import Trace
from agent.contracts import AgentPlan, PlannedTask
from agent.evidence import EvidenceAggregator

# ---------------------------------------------------------------------------
# Cross-image tasks
# ---------------------------------------------------------------------------

# Tasks that require two images to be spatially aligned before inference.
_CROSS_IMAGE_TASKS = (
    TaskType.CHANGE_DETECTION,
    TaskType.CHANGE_VQA,
    TaskType.OPTICAL_SAR_FUSION,
)


# ---------------------------------------------------------------------------
# Part 3 / Part 4 / Part 5 callable contracts
# ---------------------------------------------------------------------------

TileImageFn = Callable[..., Awaitable[list[Tile]]]

CheckCoregFn = Callable[
    [str, str],
    Awaitable[CoregistrationResult],
]

RunInferenceFn = Callable[..., Awaitable[Evidence]]

ValidateRespondFn = Callable[
    [str, Evidence],
    Awaitable[FinalResponse],
]


@dataclass
class Part345Functions:
    """
    Dependency-injection bundle for planner-owned Part 3 / Part 4 / Part 5
    calls.
    """

    tile_image: TileImageFn
    check_coregistration: CheckCoregFn
    run_inference: RunInferenceFn
    validate_and_respond: ValidateRespondFn


# ---------------------------------------------------------------------------
# Runtime execution result
# ---------------------------------------------------------------------------

@dataclass
class TaskExecution:
    """
    Runtime result produced by one planned specialist task.
    """

    task_id: str
    evidence: Evidence


# ---------------------------------------------------------------------------
# Special responses
# ---------------------------------------------------------------------------

def _unsupported_response(reason: str) -> FinalResponse:
    """
    Build a deterministic response for an unsupported request.
    """

    evidence = Evidence(
        task=TaskType.UNSUPPORTED,
        model_used="none",
        modality_used=[],
        confidence=Confidence(
            value=None,
            band="LOW",
            basis="no matching capability",
        ),
    )

    return FinalResponse(
        answer_text=f"I can't help with that: {reason}",
        confidence=evidence.confidence,
        evidence=evidence,
        abstained=True,
        abstain_reason=reason,
    )


def _refusal_evidence(
    task: TaskType,
    images: list[ImageMetadata],
    reason: str,
) -> Evidence:
    """
    Build Evidence for a co-registration refusal.
    """

    return Evidence(
        task=task,
        model_used="none",
        modality_used=[img.modality for img in images],
        confidence=Confidence(
            value=None,
            band="LOW",
            basis="co-registration check failed",
        ),
        warnings=[f"co-registration refusal: {reason}"],
    )


# ---------------------------------------------------------------------------
# Agent planning
# ---------------------------------------------------------------------------

def build_plan(
    task: TaskType,
    query: str,
    images: list[ImageMetadata],
) -> AgentPlan:
    """
    Build an explicit multi-task agent plan.

    The primary task is preserved from the upstream intent classifier.
    Additional specialist tasks are added when the query and available
    image modalities indicate that extra evidence is required.

    Current planning signals:
    - spatial language -> GROUNDING
    - temporal/change language + 2 images -> CHANGE_DETECTION
    - optical + SAR pair -> OPTICAL_SAR_FUSION

    This remains deterministic for now. A semantic/LLM planner can later
    replace this decomposition without changing the execution contract.
    """

    image_ids = [
        image.image_id
        for image in images
    ]

    tasks: list[PlannedTask] = []

    # -------------------------------------------------------------------
    # Primary task
    # -------------------------------------------------------------------

    primary_task = PlannedTask(
        task_id="task_1",
        task_type=task,
        specialist="auto",
        image_ids=image_ids,
    )

    tasks.append(primary_task)

    # -------------------------------------------------------------------
    # Query signals
    # -------------------------------------------------------------------

    query_lower = query.lower()

    grounding_words = (
        "where",
        "location",
        "highlight",
        "locate",
        "area",
        "region",
        "which area",
    )

    change_words = (
        "change",
        "changed",
        "changes",
        "difference",
        "differences",
        "before and after",
        "before-after",
        "over time",
        "temporal",
    )

    needs_grounding = any(
        word in query_lower
        for word in grounding_words
    )

    needs_change = any(
        word in query_lower
        for word in change_words
    )

    # -------------------------------------------------------------------
    # Image modality signals
    # -------------------------------------------------------------------

    modalities = {
        image.modality
        for image in images
    }

    has_optical = any(
        image.modality.value == "OPTICAL"
        for image in images
    )

    has_sar = any(
        image.modality.value == "SAR"
        for image in images
    )

    has_optical_sar_pair = (
        len(images) == 2
        and has_optical
        and has_sar
    )

    # -------------------------------------------------------------------
    # Additional change specialist
    # -------------------------------------------------------------------

    if (
        needs_change
        and len(images) == 2
        and task not in (
            TaskType.CHANGE_DETECTION,
            TaskType.CHANGE_VQA,
        )
    ):
        tasks.append(
            PlannedTask(
                task_id=f"task_{len(tasks) + 1}",
                task_type=TaskType.CHANGE_DETECTION,
                specialist="change_detection",
                image_ids=image_ids,
                depends_on=["task_1"],
                parameters={
                    "reason": "temporal/change evidence requested",
                },
            )
        )

    # -------------------------------------------------------------------
    # Additional optical + SAR specialist
    # -------------------------------------------------------------------

    if (
        has_optical_sar_pair
        and task != TaskType.OPTICAL_SAR_FUSION
    ):
        tasks.append(
            PlannedTask(
                task_id=f"task_{len(tasks) + 1}",
                task_type=TaskType.OPTICAL_SAR_FUSION,
                specialist="optical_sar_fusion",
                image_ids=image_ids,
                depends_on=["task_1"],
                parameters={
                    "reason": "optical + SAR cross-modal evidence requested",
                },
            )
        )

    # -------------------------------------------------------------------
    # Additional spatial specialist
    # -------------------------------------------------------------------
    # -----------------------------------------------------------------------

    if (
        needs_grounding
        and task != TaskType.GROUNDING
        and len(images) >= 1
    ):
        tasks.append(
            PlannedTask(
                task_id=f"task_{len(tasks) + 1}",
                task_type=TaskType.GROUNDING,
                specialist="grounding",
                image_ids=image_ids,
                depends_on=[
                    next(
                        (
                            planned_task.task_id
                            for planned_task in tasks
                            if planned_task.task_type == TaskType.CHANGE_DETECTION
                        ),
                        "task_1",
                    )
                ],
                parameters={
                    "reason": "spatial evidence requested",
                },
            )
        )

    return AgentPlan(
        intent=query,
        tasks=tasks,
        final_strategy="aggregate_and_verify",
        confidence=None,
    )

# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class Planner:
    async def execute(
        self,
        task: TaskType,
        query: str,
        images: list[ImageMetadata],
        trace: Trace,
        inference_gate: asyncio.Semaphore,
        funcs: Part345Functions,
    ) -> FinalResponse:
        """
        Execute every task in the AgentPlan.

        Current execution model:

            User query
                 ↓
            AgentPlan
                 ↓
            PlannedTask 1
                 ↓
              Evidence
                 ↓
            PlannedTask 2
                 ↓
              Evidence
                 ↓
            Part 5
                 ↓
            FinalResponse

        The execution context is intentionally kept local for now.

        Later we will introduce a proper evidence aggregation layer so that
        outputs from multiple specialists can be combined and verified
        before the final response is generated.
        """

        # -------------------------------------------------------------------
        # 1. Build the explicit agent plan
        # -------------------------------------------------------------------

        plan = build_plan(
            task,
            query,
            images,
        )

        # -------------------------------------------------------------------
        # 2. Unsupported task: deterministic early exit
        # -------------------------------------------------------------------

        if task == TaskType.UNSUPPORTED:
            return _unsupported_response(
                "this doesn't match a supported analysis for the image(s) "
                "and question given"
            )

        # -------------------------------------------------------------------
        # 3. Execute every planned task
        # -------------------------------------------------------------------

        executions: list[TaskExecution] = []

        for planned_task in plan.tasks:

            # ---------------------------------------------------------------
            # 3a. Cross-image compatibility / co-registration gate
            # ---------------------------------------------------------------

            if (
                planned_task.task_type in _CROSS_IMAGE_TASKS
                and len(images) == 2
            ):
                async with trace.stage(
                    "coregistration_check"
                ):
                    coreg = await funcs.check_coregistration(
                        images[0].image_id,
                        images[1].image_id,
                    )

                if not coreg.aligned:
                    evidence = _refusal_evidence(
                        planned_task.task_type,
                        images,
                        coreg.reason or "offset too large",
                    )

                    executions.append(
                        TaskExecution(
                            task_id=planned_task.task_id,
                            evidence=evidence,
                        )
                    )

                    async with trace.stage(
                        "response_generation"
                    ):
                        return await funcs.validate_and_respond(
                            query,
                            evidence,
                        )

            # ---------------------------------------------------------------
            # 3b. Tile input images through Part 3
            # ---------------------------------------------------------------

            all_tiles: list[Tile] = []

            async with trace.stage(
                "tiling"
            ):
                for img in images:
                    all_tiles.extend(
                        await funcs.tile_image(
                            img.image_id,
                            planned_task.task_type,
                            1024,
                            0.15,
                        )
                    )

            # ---------------------------------------------------------------
            # 3c. Build image context for Part 4
            # ---------------------------------------------------------------

            image_modalities = {
                img.image_id: img.modality
                for img in images
            }

            image_order = [
                img.image_id
                for img in images
            ]

            # ---------------------------------------------------------------
            # 3d. Execute specialist inference
            # ---------------------------------------------------------------

            async with trace.stage(
                "inference"
            ):
                async with inference_gate:
                    evidence = await funcs.run_inference(
                        planned_task.task_type,
                        all_tiles,
                        query=query,
                        image_modalities=image_modalities,
                        image_order=image_order,
                    )

            # ---------------------------------------------------------------
            # 3e. Store specialist evidence
            # ---------------------------------------------------------------

            executions.append(
                TaskExecution(
                    task_id=planned_task.task_id,
                    evidence=evidence,
                )
            )

        # -------------------------------------------------------------------
        # 4. Ensure at least one task produced evidence
        # -------------------------------------------------------------------

        if not executions:
            return _unsupported_response(
                "the planner produced no executable analysis task"
            )


        # -------------------------------------------------------------------
        # 5. Aggregate specialist evidence
        # -------------------------------------------------------------------

        evidence_objects = [
            execution.evidence
            for execution in executions
        ]

        aggregator = EvidenceAggregator()

        async with trace.stage("evidence_aggregation"):
            aggregated_evidence = aggregator.aggregate(
                evidence_objects
            )

        # -------------------------------------------------------------------
        # 6. Generate final response through Part 5
        # -------------------------------------------------------------------

        async with trace.stage(
            "response_generation"
        ):
            return await funcs.validate_and_respond(
                query,
                aggregated_evidence,
            )