"""
Proves Definition of Done #3 (Section 3.4): "VRAM stays within a configured
budget across a session using 2+ models." Uses a fake model_factory so this
runs with zero GPU and zero torch dependency.
"""
from __future__ import annotations

import pytest

from backend.model_registry.loader import ModelLoader
from backend.shared.schemas import Modality, ModelRegistryEntry, TaskType


class FakeHandle:
    def __init__(self, name: str, vram_mb: float):
        self.name = name
        self.vram_mb = vram_mb
        self.closed = False

    def close(self):
        self.closed = True


def make_entry(name: str, quantization: str = "none") -> ModelRegistryEntry:
    return ModelRegistryEntry(
        name=name, version="0.0.1",
        tasks=[TaskType.CAPTIONING], modalities=[Modality.OPTICAL],
        checkpoint_path="(test)", quantization=quantization,
        requires_coregistration=False, max_input_px=1024,
    )


@pytest.fixture
def fake_factory():
    """Returns (factory, call_log). vram_mb per (name, quant) is controllable via VRAM_BY_QUANT."""
    call_log = []
    vram_by_quant = {"none": 5000.0, "8bit": 2500.0, "4bit": 1200.0}

    def factory(entry, device, quantization_override):
        quant = quantization_override or entry.quantization
        call_log.append((entry.name, quant))
        return FakeHandle(entry.name, vram_by_quant[quant])

    return factory, call_log


def test_lazy_load_only_happens_once_per_model(fake_factory):
    factory, call_log = fake_factory
    entry_a = make_entry("model-a")
    loader = ModelLoader([entry_a], vram_budget_mb=100_000, model_factory=factory)

    loader.get("model-a")
    loader.get("model-a")
    loader.get("model-a")

    assert call_log == [("model-a", "none")]
    assert loader.resident_models() == ["model-a"]


def test_lru_eviction_when_budget_exceeded(fake_factory):
    factory, call_log = fake_factory
    entry_a, entry_b = make_entry("model-a"), make_entry("model-b")
    # Budget fits exactly one 5000 MB model at a time.
    loader = ModelLoader([entry_a, entry_b], vram_budget_mb=6000.0, model_factory=factory)

    loader.get("model-a")
    assert loader.resident_models() == ["model-a"]

    loader.get("model-b")
    assert loader.resident_models() == ["model-b"], "model-a should have been evicted to fit model-b"
    assert loader.vram_used_mb() <= 6000.0


def test_touching_a_model_protects_it_from_eviction(fake_factory):
    factory, _ = fake_factory
    entry_a, entry_b, entry_c = make_entry("model-a"), make_entry("model-b"), make_entry("model-c")
    # Budget fits two models but not three.
    loader = ModelLoader([entry_a, entry_b, entry_c], vram_budget_mb=11_000.0, model_factory=factory)

    loader.get("model-a")
    loader.get("model-b")
    loader.get("model-a")  # touch A again -> B is now the least-recently-used
    loader.get("model-c")  # should evict B, not A

    assert "model-a" in loader.resident_models()
    assert "model-b" not in loader.resident_models()
    assert "model-c" in loader.resident_models()


def test_quantization_override_reloads_without_double_counting_vram(fake_factory):
    factory, call_log = fake_factory
    entry_a = make_entry("model-a", quantization="none")
    loader = ModelLoader([entry_a], vram_budget_mb=100_000, model_factory=factory)

    loader.get("model-a")  # loads at "none" (5000 MB)
    assert loader.vram_used_mb() == 5000.0

    loader.get("model-a", quantization_override="8bit")  # OOM retry ladder asking for a lighter copy

    assert call_log == [("model-a", "none"), ("model-a", "8bit")]
    assert loader.resident_models() == ["model-a"], "should still be exactly one resident copy"
    assert loader.vram_used_mb() == 2500.0, "stale 'none' copy must not still be counted"


def test_single_oversized_model_loads_anyway_as_best_effort(fake_factory):
    factory, _ = fake_factory
    entry_a = make_entry("model-a")  # 5000 MB
    loader = ModelLoader([entry_a], vram_budget_mb=100.0, model_factory=factory)  # budget far too small

    handle = loader.get("model-a")  # should not raise

    assert handle.name == "model-a"
    assert loader.resident_models() == ["model-a"]


def test_unload_calls_close_on_the_handle(fake_factory):
    factory, _ = fake_factory
    entry_a, entry_b = make_entry("model-a"), make_entry("model-b")
    loader = ModelLoader([entry_a, entry_b], vram_budget_mb=6000.0, model_factory=factory)

    loader.get("model-a")
    handle_a = loader._resident["model-a"].handle
    loader.get("model-b")  # evicts model-a

    assert handle_a.closed is True


def test_health_reports_budget_and_usage(fake_factory):
    factory, _ = fake_factory
    entry_a = make_entry("model-a")
    loader = ModelLoader([entry_a], vram_budget_mb=8192.0, model_factory=factory)
    loader.get("model-a")

    health = loader.health()

    assert health["models_loaded"] == ["model-a"]
    assert health["vram_used_mb"] == 5000.0
    assert health["vram_budget_mb"] == 8192.0
