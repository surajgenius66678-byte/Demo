"""
Proves Definition of Done #2 (Section 3.4): "survives a simulated OOM
without crashing the process." Section 3.4's hardening list: "try/except
around every model call catching OOM specifically: retry smaller batch ->
smaller/quantized model -> explicit entry in Evidence.warnings, never a
raw process crash."
"""
from __future__ import annotations

import pytest

from backend.model_registry.inference import InferenceEngine, _run_with_oom_ladder
from backend.model_registry.loader import ModelLoader
from backend.shared.schemas import Confidence, Evidence, Modality, ModelRegistryEntry, TaskType


def make_entry(quantization: str = "none") -> ModelRegistryEntry:
    return ModelRegistryEntry(
        name="oom-test-model", version="0.0.1",
        tasks=[TaskType.SINGLE_IMAGE_VQA, TaskType.CAPTIONING], modalities=[Modality.OPTICAL],
        checkpoint_path="(test)", quantization=quantization,
        requires_coregistration=False, max_input_px=1024,
    )


def fake_evidence(batch, quant) -> Evidence:
    return Evidence(
        task=TaskType.CAPTIONING, model_used="oom-test-model", modality_used=[Modality.OPTICAL],
        vqa_answer_raw=f"ok:{len(batch)}@{quant}",
        confidence=Confidence(value=0.9, band="HIGH", basis="fake"),
    )


# --- Pure ladder logic, isolated from adapters/loader entirely ---

def test_ladder_shrinks_batch_before_touching_quantization(four_tiles):
    entry = make_entry()
    calls = []

    def process_batch(batch, quant):
        calls.append((len(batch), quant))
        if len(batch) > 1:
            raise RuntimeError("CUDA out of memory")
        return fake_evidence(batch, quant)

    partial, warnings = _run_with_oom_ladder(four_tiles, entry, process_batch)

    assert len(partial) == 4, "every tile should eventually succeed at batch size 1"
    assert all(quant == "none" for _, quant in calls), "should never have needed to touch quantization"
    assert any("OOM" in w for w in warnings)


def test_ladder_falls_back_to_quantization_after_batch_size_hits_one(four_tiles):
    entry = make_entry()
    calls = []

    def process_batch(batch, quant):
        calls.append((len(batch), quant))
        if quant == "none":
            raise RuntimeError("CUDA out of memory")
        return fake_evidence(batch, quant)

    partial, warnings = _run_with_oom_ladder(four_tiles, entry, process_batch)

    assert any(quant == "8bit" for _, quant in calls), "should have stepped down to 8bit after batch_size=1 still OOM'd"
    # After a quant step-down the ladder resets batch_size to the full
    # remaining set and retries in one shot (a lighter model can afford a
    # bigger batch) -- so this resolves as ONE batch of all 4 tiles at
    # 8bit, not 4 separate single-tile batches like the pure batch-shrink
    # case above. Both are correct; they're just different recovery paths.
    assert len(partial) == 1
    assert partial[0].vqa_answer_raw == "ok:4@8bit"


def test_ladder_never_raises_even_when_everything_always_ooms(four_tiles):
    entry = make_entry()

    def always_oom(batch, quant):
        raise RuntimeError("CUDA out of memory: always fails")

    # Must not raise -- this is the literal DoD requirement.
    partial, warnings = _run_with_oom_ladder(four_tiles, entry, always_oom)

    assert partial == [], "nothing could be produced, but that's still not a crash"
    assert len(warnings) > 0
    assert any("Dropping tile" in w for w in warnings)


def test_ladder_propagates_non_oom_exceptions(four_tiles):
    entry = make_entry()

    def broken(batch, quant):
        raise KeyError("this is a real bug, not OOM")

    with pytest.raises(KeyError):
        _run_with_oom_ladder(four_tiles, entry, broken)


def test_engine_run_inference_returns_degraded_evidence_not_exception_on_total_oom(single_tile, sample_registry):
    """End-to-end: even when the model factory itself always OOMs, run_inference
    must return a schema-valid (if degraded) Evidence, never let the exception escape."""

    def always_oom_factory(entry, device, quantization_override):
        class _Handle:
            def generate(self, image_path, prompt):
                raise RuntimeError("CUDA out of memory")
        return _Handle()

    loader = ModelLoader(sample_registry, model_factory=always_oom_factory)
    engine = InferenceEngine(sample_registry, loader)

    evidence = engine.run_inference(TaskType.CAPTIONING, [single_tile])

    assert isinstance(evidence, Evidence)
    assert evidence.confidence.band == "LOW"
    assert len(evidence.warnings) > 0


def test_engine_run_inference_succeeds_after_transient_oom(single_tile, sample_registry):
    """The model OOMs at full precision but works at 8bit -- proves the quant
    fallback actually reaches a working model, not just that it doesn't crash."""

    class _Handle:
        def __init__(self, quant):
            self.quant = quant

        def generate(self, image_path, prompt):
            if self.quant == "none":
                raise RuntimeError("CUDA out of memory")
            return {"text": f"described at {self.quant}", "score": 0.6}

    def factory(entry, device, quantization_override):
        return _Handle(quantization_override or entry.quantization)

    loader = ModelLoader(sample_registry, model_factory=factory)
    engine = InferenceEngine(sample_registry, loader)

    evidence = engine.run_inference(TaskType.CAPTIONING, [single_tile])

    assert evidence.vqa_answer_raw == "described at 8bit"
    assert any("OOM" in w for w in evidence.warnings)
