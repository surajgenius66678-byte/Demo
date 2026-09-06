"""
Part 4 — model registry.

Config-driven list of ModelRegistryEntry (Section 4 schema). Nothing in
this file touches a GPU or imports torch/transformers, so it's fully
testable without any ML dependencies installed.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from backend.model_registry.exceptions import ModelNotFoundError
from backend.shared.schemas import Modality, ModelRegistryEntry, TaskType

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "models.yaml"


def load_registry_config(path: str | Path = DEFAULT_CONFIG_PATH) -> list[ModelRegistryEntry]:
    """Parse models.yaml into a list of validated ModelRegistryEntry objects."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    entries_raw = raw.get("models", []) if raw else []
    return [ModelRegistryEntry(**entry) for entry in entries_raw]


def find_models_for_task(
    registry: list[ModelRegistryEntry],
    task: TaskType,
    modalities: list[Modality] | None = None,
) -> list[ModelRegistryEntry]:
    """
    Every registry entry whose `tasks` includes `task`, optionally filtered
    to entries that support every modality in `modalities`. Config order is
    preserved, so config order doubles as priority order among ties.
    """
    candidates = [e for e in registry if task in e.tasks]
    if modalities:
        needed = set(modalities)
        candidates = [e for e in candidates if needed.issubset(set(e.modalities))]
    return candidates


def select_model(
    registry: list[ModelRegistryEntry],
    task: TaskType,
    modalities: list[Modality] | None = None,
    model_hint: str | None = None,
) -> ModelRegistryEntry:
    """
    Resolve (task, modalities, optional explicit model_hint) to exactly one
    ModelRegistryEntry. model_hint, if given, must name a registered model
    that actually supports the task — it narrows which matching entry gets
    picked, it does not bypass the capability check.
    """
    candidates = find_models_for_task(registry, task, modalities)
    if model_hint:
        candidates = [e for e in candidates if e.name == model_hint]
    if not candidates:
        raise ModelNotFoundError(
            f"No registered model serves task={task.value!r} "
            f"modalities={[m.value for m in modalities] if modalities else 'any'} "
            f"model_hint={model_hint!r}"
        )
    return candidates[0]
