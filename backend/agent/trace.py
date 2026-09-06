"""
Audit-trail recording for a single job.

Section 3.2 requires "audit-trail logging per stage (name, status, duration)"
and the HTTP contract for GET /api/jobs/{id}/trace is:

    {"steps": [{"stage": string, "status": string, "duration_ms": number}]}

`Trace.stage(name)` is an async context manager: wrap every call out to
Part 3/4/5 in it and a step is appended automatically, whether that call
succeeds or raises. It does not swallow the exception — it records the
failure and re-raises, so JobQueue._run() (see queue.py) is the single
place that decides what a failure means for overall job status.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field


@dataclass
class TraceStep:
    stage: str
    status: str  # "success" | "failed"
    duration_ms: float

    def to_dict(self) -> dict:
        return {"stage": self.stage, "status": self.status, "duration_ms": round(self.duration_ms, 2)}


@dataclass
class Trace:
    steps: list[TraceStep] = field(default_factory=list)

    @asynccontextmanager
    async def stage(self, name: str):
        start = time.monotonic()
        try:
            yield
        except Exception:
            self.steps.append(TraceStep(name, "failed", (time.monotonic() - start) * 1000))
            raise
        else:
            self.steps.append(TraceStep(name, "success", (time.monotonic() - start) * 1000))

    def as_dict(self) -> dict:
        return {"steps": [s.to_dict() for s in self.steps]}
