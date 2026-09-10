from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from shared.schemas import Evidence, TaskType


@dataclass
class PlannedTask:
    task_id: str
    task_type: TaskType
    specialist: str
    image_ids: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskExecution:
    task_id: str
    evidence: Evidence
    input_task_ids: list[str] = field(default_factory=list)
    input_evidence: list[Evidence] = field(default_factory=list)

@dataclass
class AgentPlan:
    intent: str
    tasks: list[PlannedTask] = field(default_factory=list)
    final_strategy: str = "aggregate_and_verify"
    confidence: Optional[float] = None