from backend.evidence.claim_check import check_numeric_claims
from backend.evidence.grounding import build_grounding_context
from backend.evidence.service import validate_and_respond
from backend.shared.schemas import FinalResponse, TaskType
from tests.evidence import fixtures


def test_every_task_type_produces_schema_valid_response_with_grounded_numbers():
    """Definition of done, part 1: schema-valid FinalResponse for a fixture
    battery covering every task type; no number in answer_text is ever
    untraceable to the evidence dict."""
    for task, factory in fixtures.ALL_TASK_FIXTURES.items():
        evidence = factory()
        response = validate_and_respond(f"query for {task.value}", evidence)
        assert isinstance(response, FinalResponse)
        assert response.answer_text

        ctx = build_grounding_context(evidence)
        check = check_numeric_claims(response.answer_text, ctx)
        assert check.ok, (
            f"{task}: untraceable numbers {check.unverified_numbers} "
            f"in: {response.answer_text!r}"
        )

        if task == TaskType.UNSUPPORTED:
            assert response.abstained
            assert response.confidence.value is None


def test_weak_and_empty_evidence_abstain_with_grounded_response():
    for evidence in (fixtures.weak_empty(), fixtures.weak_low_confidence(), fixtures.coregistration_refused()):
        response = validate_and_respond("some query", evidence)
        assert response.abstained
        assert response.abstain_reason
        assert response.answer_text  # never an empty string
        assert response.confidence.value is None
        assert response.confidence.band == "LOW"


def test_response_always_carries_the_original_evidence_through():
    evidence = fixtures.change_detection()
    response = validate_and_respond("what changed", evidence)
    assert response.evidence is evidence


def test_positional_call_matches_the_architecture_doc_contract():
    """Part 2 calls this as validate_and_respond(query, evidence) with no
    keyword args — confirm that exact call shape still works."""
    response = validate_and_respond("describe this image", fixtures.single_image_vqa())
    assert isinstance(response, FinalResponse)


def test_warning_numbers_do_not_discard_a_good_llm_answer():
    # Regression (end-to-end): before grounding.py scanned evidence.warnings
    # for numbers, service.py's outer safety-net re-check would see the
    # warning's numbers as "unverified" and silently swap out a perfectly
    # good LLM-generated answer for the generic template — even though
    # nothing was actually wrong with the LLM's answer.
    from backend.shared.schemas import ChangeMap, Confidence, Evidence, Modality, TaskType

    class FakeClient:
        def generate(self, system_prompt, user_prompt):
            return "About {changed_area_pct}% of the area changed — a nice natural sentence."

    ev = Evidence(
        task=TaskType.CHANGE_DETECTION,
        model_used="m",
        modality_used=[Modality.OPTICAL],
        change_map=ChangeMap(probability_raster_path="p", changed_area_px=1000, changed_area_pct=10.0, mean_confidence=0.6),
        confidence=Confidence(value=0.6, band="MEDIUM", basis="x"),
        warnings=["retried after OOM: batch size reduced from 32 to 8"],
    )
    response = validate_and_respond("what changed", ev, llm_client=FakeClient())
    assert "nice natural sentence" in response.answer_text
    assert "batch size reduced from 32 to 8" in response.answer_text
