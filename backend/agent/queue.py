"""
JobQueue owns the three queue-related hardening items from Section 3.2:

  - "A single bounded-concurrency job queue"
  - "Bounded-concurrency queue around every Part 4 (GPU) call — two
    simultaneous requests must queue, never collide"
  - "Job execution decoupled from the HTTP connection — closing the tab
    must not kill the job"
  - "try/except around every Part 3/4/5 call — one failed job must never
    take the process down"

submit() only registers a JobRecord and schedules a background asyncio task
— it never awaits the job itself, so the HTTP handler that calls it (POST
/api/query in api/main.py) returns immediately regardless of how long the
job takes or whether the client is still connected. The concurrency bound
on run_inference lives in the asyncio.Semaphore passed into Planner.execute
(see planner.py) — this class owns creating and sharing that one semaphore
across every job, which is what actually makes "two simultaneous requests
queue instead of colliding" true.

Trace-step-level failures are logged and re-raised by Trace.stage() (see
trace.py); _run() is the single place that turns an exception, wherever it
came from, into job status "failed" instead of letting it propagate into
asyncio's default (silent, process-surviving but easy to lose track of)
unhandled-task-exception handling.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Optional

from shared.schemas import FinalResponse, ImageMetadata, TaskType
from agent.intent import classify_intent
from agent.planner import Part345Functions, Planner
from agent.trace import Trace


@dataclass
class JobRecord:
    job_id: str
    status: str = "queued"  # "queued" | "running" | "done" | "failed"
    progress: list[str] = field(default_factory=list)
    result: Optional[FinalResponse] = None
    error: Optional[str] = None
    trace: Trace = field(default_factory=Trace)


class JobQueue:
    def __init__(self, funcs: Part345Functions, max_concurrent_inference: int = 1):
        self._jobs: dict[str, JobRecord] = {}
        self._funcs = funcs
        self._inference_gate = asyncio.Semaphore(max_concurrent_inference)
        self._planner = Planner()
        self._background_tasks: set[asyncio.Task] = set()

    def submit(self, query: str, images: list[ImageMetadata]) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = JobRecord(job_id=job_id)
        task = asyncio.create_task(self._run(job_id, query, images))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return job_id

    def get(self, job_id: str) -> Optional[JobRecord]:
        return self._jobs.get(job_id)

    async def _run(self, job_id: str, query: str, images: list[ImageMetadata]) -> None:
        record = self._jobs[job_id]
        record.status = "running"
        try:
            record.progress.append("classifying intent")
            task_type: TaskType = await classify_intent(query, images)
            record.progress.append(f"classified as {task_type.value}")

            result = await self._planner.execute(
                task_type, query, images, record.trace, self._inference_gate, self._funcs
            )

            record.result = result
            record.status = "done"
            record.progress.append("done")
        except Exception as e:  # one failed job must never take the process down
            record.status = "failed"
            record.error = str(e)
            record.progress.append("failed")
