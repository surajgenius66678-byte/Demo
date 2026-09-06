from backend.evidence.validator import validate_evidence
from backend.shared.schemas import TaskType
from tests.evidence import fixtures


def test_unsupported_task_abstains():
    result = validate_evidence("what is the population here", fixtures.unsupported())
    assert result.should_abstain
    assert result.reason


def test_coregistration_refusal_abstains():
    result = validate_evidence("what changed", fixtures.coregistration_refused())
    assert result.should_abstain
    assert "align" in result.reason.lower() or "coregist" in result.reason.lower()


def test_empty_evidence_abstains():
    result = validate_evidence("describe this image", fixtures.weak_empty())
    assert result.should_abstain


def test_weak_low_confidence_abstains():
    result = validate_evidence("find buildings", fixtures.weak_low_confidence())
    assert result.should_abstain


def test_zero_detections_with_raw_answer_is_not_treated_as_empty():
    from backend.shared.schemas import Confidence, Evidence, Modality

    ev = Evidence(
        task=TaskType.GROUNDING,
        model_used="grounding-dino-rs",
        modality_used=[Modality.OPTICAL],
        detections=[],
        vqa_answer_raw="No buildings were detected in this area.",
        confidence=Confidence(value=0.6, band="MEDIUM", basis="absence confidence"),
    )
    result = validate_evidence("find buildings", ev)
    assert not result.should_abstain


def test_healthy_evidence_does_not_abstain():
    for task, factory in fixtures.ALL_TASK_FIXTURES.items():
        if task == TaskType.UNSUPPORTED:
            continue
        result = validate_evidence("test query", factory())
        assert not result.should_abstain, f"{task} unexpectedly abstained: {result.reason}"
