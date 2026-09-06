"""
Planner.execute() sequences Part 3 -> Part 4 -> Part 5 calls for a single
classified task (Section 3.2: "A planner that sequences Part 3 -> Part 4 ->
Part 5 calls per task type").

Two deliberate design choices beyond the literal endpoint/function list:

1. Co-registration gate. Section 3.3's hardening says check_coregistration
   "returns aligned: false above the offset threshold; Part 2 is responsible
   for refusing the downstream task on that result." Applied here to every
   task that combines two images pixel-for-pixel — CHANGE_DETECTION,
   CHANGE_VQA, and OPTICAL_SAR_FUSION — not just the change tasks, since
   fusing misaligned optical/SAR is exactly as unreliable as diffing
   misaligned before/after images. On failure, Part 2 does NOT call Part 4
   at all (that is what "refusing" means here) — it builds a minimal
   Evidence carrying the failure reason in `warnings` and routes straight to
   validate_and_respond, so Part 5's existing abstain path (Section 3.5:
   "If ... evidence.warnings contains a co-registration refusal, sets
   abstained: true") produces one consistent, well-formatted decline instead
   of Part 2 inventing a second message format.

2. UNSUPPORTED short-circuit. There is no Evidence to generate a response
   from, and Section 3.2's hardening wants "a plain-language response, not
   a forced wrong answer" — so this builds the FinalResponse directly, with
   a fixed template sentence, rather than manufacturing empty Evidence just
   to route it through Part 5. Nothing here is free-generated text, so it
   doesn't cross into Part 5's "answer-text generation" territory.

`inference_gate` is the bounded-concurrency semaphore from Section 3.2's
hardening ("Bounded-concurrency queue around every Part 4 (GPU) call"). It
is acquired around run_inference only — tiling and co-registration aren't
GPU work and queuing them too would just add latency for no benefit.
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

_CROSS_IMAGE_TASKS = (TaskType.CHANGE_DETECTION, TaskType.CHANGE_VQA, TaskType.OPTICAL_SAR_FUSION)

TileImageFn = Callable[[str, TaskType, int, float], Awaitable[list[Tile]]]
CheckCoregFn = Callable[[str, str], Awaitable[CoregistrationResult]]
RunInferenceFn = Callable[[TaskType, list[Tile], Optional[str]], Awaitable[Evidence]]
ValidateRespondFn = Callable[[str, Evidence], Awaitable[FinalResponse]]


@dataclass
class Part345Functions:
    """Dependency-injection bundle for the 4 planner-owed calls (validate_and_prepare
    is only used at upload time, in api/main.py, so it isn't part of this bundle)."""
    tile_image: TileImageFn
    check_coregistration: CheckCoregFn
    run_inference: RunInferenceFn
    validate_and_respond: ValidateRespondFn


def _unsupported_response(reason: str) -> FinalResponse:
    evidence = Evidence(
        task=TaskType.UNSUPPORTED, model_used="none", modality_used=[],
        confidence=Confidence(value=None, band="LOW", basis="no matching capability"),
    )
    return FinalResponse(
        answer_text=f"I can't help with that: {reason}",
        confidence=evidence.confidence, evidence=evidence,
        abstained=True, abstain_reason=reason,
    )


def _refusal_evidence(task: TaskType, images: list[ImageMetadata], reason: str) -> Evidence:
    return Evidence(
        task=task, model_used="none", modality_used=[img.modality for img in images],
        confidence=Confidence(value=None, band="LOW", basis="co-registration check failed"),
        warnings=[f"co-registration refusal: {reason}"],
    )


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
        if task == TaskType.UNSUPPORTED:
            return _unsupported_response(
                "this doesn't match a supported analysis for the image(s) and question given"
            )

        if task in _CROSS_IMAGE_TASKS and len(images) == 2:
            async with trace.stage("coregistration_check"):
                coreg = await funcs.check_coregistration(images[0].image_id, images[1].image_id)
            if not coreg.aligned:
                evidence = _refusal_evidence(task, images, coreg.reason or "offset too large")
                async with trace.stage("response_generation"):
                    return await funcs.validate_and_respond(query, evidence)

        all_tiles: list[Tile] = []
        async with trace.stage("tiling"):
            for img in images:
                all_tiles.extend(await funcs.tile_image(img.image_id, task))

        async with trace.stage("inference"):
            async with inference_gate:
                evidence = await funcs.run_inference(task, all_tiles)

        async with trace.stage("response_generation"):
            return await funcs.validate_and_respond(query, evidence)
