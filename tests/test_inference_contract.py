"""
Proves Section 3.2's literal contract: Part 2 imports and calls

    run_inference(task, tiles, model_hint=None) -> Evidence
    list_available_models() -> list[ModelRegistryEntry]
    health_check() -> dict

as bare module-level functions. This test suite calls them exactly the way
Part 2's code will, through the mock engine, since that's what Part 2 and
Part 5 build against before Part 6 has real checkpoints.
"""
from __future__ import annotations

from backend.model_registry import inference
from backend.shared.schemas import Evidence, ModelRegistryEntry, TaskType


def setup_function(_):
    # Fresh mock engine before each test -- avoids state leaking between tests.
    inference.configure_engine(use_mock=True)


def test_run_inference_is_a_bare_module_function(single_tile):
    evidence = inference.run_inference(TaskType.CAPTIONING, [single_tile])
    assert isinstance(evidence, Evidence)


def test_run_inference_works_with_only_the_literal_3_positional_args(single_tile):
    """Section 3.2's exact signature -- no keyword extensions used."""
    evidence = inference.run_inference(TaskType.CAPTIONING, [single_tile], None)
    assert isinstance(evidence, Evidence)


def test_run_inference_accepts_model_hint(single_tile):
    evidence = inference.run_inference(TaskType.CAPTIONING, [single_tile], model_hint="mock-vlm")
    assert evidence.model_used == "mock-vlm"


def test_list_available_models_returns_registry_entries():
    models = inference.list_available_models()
    assert len(models) > 0
    assert all(isinstance(m, ModelRegistryEntry) for m in models)


def test_health_check_returns_dict_with_status():
    health = inference.health_check()
    assert isinstance(health, dict)
    assert "status" in health


def test_importing_inference_module_never_requires_a_gpu(single_tile):
    """get_engine() must default to the mock engine so Part 2/5 developers
    never hit an ImportError for torch just from importing this module."""
    inference._engine = None  # simulate a fresh process that never called configure_engine
    evidence = inference.run_inference(TaskType.CAPTIONING, [single_tile])
    assert isinstance(evidence, Evidence)
