"""
Planner.execute() sequences Part 3 -> Part 4 -> Part 5 calls for a single
classified task (Section 3.2: "A planner that sequences Part 3 -> Part 4 ->
Part 5 calls per task type").

Current responsibilities:
1. Refuse unsupported tasks without calling downstream components.
2. Gate cross-image tasks behind co-registration.
3. Tile every input image through Part 3.
4. Run Part 4 under the bounded inference semaphore.
5. Forward query/modality/order context into Part 4.
6. Pass resulting Evidence to Part 5 for final response generation.

The current planner still executes one classified task at a time. The future
agentic planner can extend this layer to sequence multiple specialists while
keeping the same Part345Functions dependency-injection boundary.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

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


# Tasks that require two images to be spatially aligned before inference.
_CROSS_IMAGE_TASKS = (
    TaskType.CHANGE_DETECTION,
    TaskType.CHANGE_VQA,
    TaskType.OPTICAL_SAR_FUSION,
)


# ---------------------------------------------------------------------------
# Part 3 / 4 / 5 callable contracts
# ---------------------------------------------------------------------------

TileImageFn = Callable[
    [str, TaskType, int, float],
    Awaitable[list[Tile]],
]

CheckCoregFn = Callable[
    [str, str],
    Awaitable[CoregistrationResult],
]

# Part 4 already supports additional keyword-only context:
#   query
#   image_modalities
#   image_order
#
# Keep this broad so the planner can pass that context without forcing every
# mock/test implementation to reproduce the exact concrete signature.
RunInferenceFn = Callable[..., Awaitable[Evidence]]

ValidateRespondFn = Callable[
    [str, Evidence],
    Awaitable[FinalResponse],
]


@dataclass
class Part345Functions:
    """
    Dependency-injection bundle for planner-owned Part 3/4/5 calls.

    validate_and_prepare is intentionally not included because it happens at
    upload time in api/main.py, before a job enters the planner.
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

    This intentionally bypasses Part 5 because there is no valid Evidence
    from which to generate a grounded answer.
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

    Part 5 already knows how to convert the special warning into an abstained
    final response, so this keeps refusal formatting centralized there.
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
        Execute the current single-task Part 3 -> Part 4 -> Part 5 pipeline.

        Important:
        - The inference semaphore surrounds only Part 4 because that is the
          bounded GPU work.
        - Query/modality/order context is forwarded to Part 4 so adapters such
          as VQA, change detection, and optical+SAR fusion receive the
          information they need.
        """

        # ---------------------------------------------------------------
        # 1. Unsupported task: deterministic early exit
        # ---------------------------------------------------------------
        if task == TaskType.UNSUPPORTED:
            return _unsupported_response(
                "this doesn't match a supported analysis for the image(s) "
                "and question given"
            )

        # ---------------------------------------------------------------
        # 2. Cross-image compatibility / co-registration gate
        # ---------------------------------------------------------------
        if task in _CROSS_IMAGE_TASKS and len(images) == 2:
            async with trace.stage("coregistration_check"):
                coreg = await funcs.check_coregistration(
                    images[0].image_id,
                    images[1].image_id,
                )

            if not coreg.aligned:
                evidence = _refusal_evidence(
                    task,
                    images,
                    coreg.reason or "offset too large",
                )

                async with trace.stage("response_generation"):
                    return await funcs.validate_and_respond(
                        query,
                        evidence,
                    )

        # ---------------------------------------------------------------
        # 3. Tile all input images through Part 3
        # ---------------------------------------------------------------
        all_tiles: list[Tile] = []

        async with trace.stage("tiling"):
            for img in images:
                all_tiles.extend(
                    await funcs.tile_image(
                        img.image_id,
                        task,
                        1024,
                        0.15,
                    )
                )

        # ---------------------------------------------------------------
        # 4. Run Part 4 under the bounded inference semaphore
        # ---------------------------------------------------------------
        image_modalities = {
            img.image_id: img.modality
            for img in images
        }

        # Preserve the submitted image order. For bi-temporal tasks this is
        # currently the best available representation of before -> after;
        # Part 2 can later become stricter once timestamp ordering is made
        # explicit in the schema.
        image_order = [
            img.image_id
            for img in images
        ]

        async with trace.stage("inference"):
            async with inference_gate:
                evidence = await funcs.run_inference(
                    task,
                    all_tiles,
                    query=query,
                    image_modalities=image_modalities,
                    image_order=image_order,
                )

        # ---------------------------------------------------------------
        # 5. Grounded response generation through Part 5
        # ---------------------------------------------------------------
        async with trace.stage("response_generation"):
            return await funcs.validate_and_respond(
                query,
                evidence,
            )