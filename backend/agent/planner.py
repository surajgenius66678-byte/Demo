"""
Planner.execute() sequences Part 3 -> Part 4 -> Part 5 calls for the
currently planned tasks.

The planner first builds an explicit AgentPlan. The execution layer then
walks through every PlannedTask, executes its specialist, stores the
resulting Evidence, and finally passes the aggregated evidence to Part 5.

The planner validates cross-image requirements before specialist inference
and preserves upstream evidence between dependent tasks.
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

from agent.contracts import AgentPlan, PlannedTask, TaskExecution
from agent.evidence import EvidenceAggregator
from agent.trace import Trace


# ---------------------------------------------------------------------------
# Cross-image tasks
# ---------------------------------------------------------------------------

# Tasks that require exactly two spatially compatible images.
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
    Build Evidence for a cross-image validation refusal.
    """

    return Evidence(
        task=task,
        model_used="none",
        modality_used=[img.modality for img in images],
        confidence=Confidence(
            value=None,
            band="LOW",
            basis="cross-image validation failed",
        ),
        warnings=[f"cross-image refusal: {reason}"],
    )


def _refusal_response(
    evidence: Evidence,
    reason: str,
) -> FinalResponse:
    """
    Build a deterministic abstaining response for planner validation
    failures.

    Planner-level refusals must not be passed through Part 5 because the
    normal response generator may treat the refusal Evidence as valid
    analysis evidence and return abstained=False.
    """

    return FinalResponse(
        answer_text=(
            "I can't perform this analysis because "
            f"{reason}"
        ),
        confidence=evidence.confidence,
        evidence=evidence,
        abstained=True,
        abstain_reason=reason,
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

    Planning remains deterministic for now so that the execution trace is
    predictable and auditable.
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

    if (
        needs_grounding
        and task != TaskType.GROUNDING
        and len(images) >= 1
    ):
        grounding_dependency = next(
            (
                planned_task.task_id
                for planned_task in tasks
                if planned_task.task_type == TaskType.CHANGE_DETECTION
            ),
            "task_1",
        )

        tasks.append(
            PlannedTask(
                task_id=f"task_{len(tasks) + 1}",
                task_type=TaskType.GROUNDING,
                specialist="grounding",
                image_ids=image_ids,
                depends_on=[grounding_dependency],
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

        Execution flow:

            User query
                 ↓
            AgentPlan
                 ↓
            PlannedTask
                 ↓
            Part 3
                 ↓
            Part 4
                 ↓
              Evidence
                 ↓
            Evidence aggregation
                 ↓
            Part 5
                 ↓
            FinalResponse
        """

        # -------------------------------------------------------------------
        # 1. Build explicit agent plan
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
            # 3a. Cross-image validation / co-registration gate
            # ---------------------------------------------------------------

            if planned_task.task_type in _CROSS_IMAGE_TASKS:

                # SIH requirement:
                # cross-image tasks require exactly two images.
                if len(images) != 2:
                    reason = (
                        f"{planned_task.task_type.value} requires exactly "
                        f"2 images, but received {len(images)}."
                    )

                    evidence = _refusal_evidence(
                        planned_task.task_type,
                        images,
                        reason,
                    )

                    executions.append(
                        TaskExecution(
                            task_id=planned_task.task_id,
                            evidence=evidence,
                        )
                    )

                    return _refusal_response(
                        evidence,
                        reason,
                    )

                # Two images are present, so verify spatial compatibility
                # before running the specialist model.
                async with trace.stage(
                    "coregistration_check"
                ):
                    coreg = await funcs.check_coregistration(
                        images[0].image_id,
                        images[1].image_id,
                    )

                if not coreg.aligned:
                    reason = coreg.reason or "offset too large"

                    evidence = _refusal_evidence(
                        planned_task.task_type,
                        images,
                        reason,
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
                        return _refusal_response(
                            evidence,
                            reason,
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
            # 3d. Resolve upstream task evidence
            # ---------------------------------------------------------------

            execution_by_id = {
                execution.task_id: execution
                for execution in executions
            }

            upstream_executions: list[TaskExecution] = []

            for dependency_id in planned_task.depends_on:
                dependency = execution_by_id.get(dependency_id)

                if dependency is None:
                    raise RuntimeError(
                        f"task {planned_task.task_id} depends on "
                        f"unfinished task {dependency_id}"
                    )

                upstream_executions.append(
                    dependency
                )

            upstream_evidence = [
                execution.evidence
                for execution in upstream_executions
            ]

            # ---------------------------------------------------------------
            # 3e. Execute specialist inference
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
                        upstream_evidence=upstream_evidence,
                    )

            # ---------------------------------------------------------------
            # 3f. Store specialist evidence
            # ---------------------------------------------------------------

            executions.append(
                TaskExecution(
                    task_id=planned_task.task_id,
                    evidence=evidence,
                    input_task_ids=[
                        execution.task_id
                        for execution in upstream_executions
                    ],
                    input_evidence=upstream_evidence,
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

        async with trace.stage(
            "evidence_aggregation"
        ):
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