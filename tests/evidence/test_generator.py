from backend.evidence.claim_check import check_numeric_claims
from backend.evidence.generator import generate_explanation
from backend.evidence.grounding import build_grounding_context
from backend.evidence.llm_client import LLMUnavailableError
from tests.evidence import fixtures


class _ScriptedLLMClient:
    """Test double: returns each string in `responses` in order, one per
    call to generate(); raises once exhausted."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        if not self._responses:
            raise LLMUnavailableError("scripted client exhausted")
        return self._responses.pop(0)


def test_no_client_uses_template_path_directly():
    ev = fixtures.change_detection()
    text = generate_explanation("what changed", ev, ev.confidence, llm_client=None)
    ctx = build_grounding_context(ev)
    assert check_numeric_claims(text, ctx).ok


def test_llm_response_using_placeholders_is_accepted_on_first_try():
    ev = fixtures.change_detection()
    client = _ScriptedLLMClient([
        "About {changed_area_pct}% of the area changed, confidence {change_mean_confidence}%.",
    ])
    text = generate_explanation("what changed", ev, ev.confidence, llm_client=client)
    assert "{changed_area_pct}" not in text  # placeholder was filled
    ctx = build_grounding_context(ev)
    assert check_numeric_claims(text, ctx).ok
    assert len(client.calls) == 1


def test_llm_leaking_a_raw_number_triggers_one_retry_then_succeeds():
    ev = fixtures.change_detection()
    client = _ScriptedLLMClient([
        "About 99% of the area changed.",  # hallucinated number
        "About {changed_area_pct}% of the area changed, corrected.",
    ])
    text = generate_explanation("what changed", ev, ev.confidence, llm_client=client)
    ctx = build_grounding_context(ev)
    assert check_numeric_claims(text, ctx).ok
    assert len(client.calls) == 2


def test_llm_repeatedly_hallucinating_falls_back_to_template():
    ev = fixtures.change_detection()
    client = _ScriptedLLMClient([
        "About 99% of the area changed.",
        "Still about 99% changed, trust me.",
    ])
    text = generate_explanation("what changed", ev, ev.confidence, llm_client=client)
    ctx = build_grounding_context(ev)
    assert check_numeric_claims(text, ctx).ok
    assert "99" not in text  # the fallback template never used the bad number
    assert len(client.calls) == 2


def test_llm_unavailable_falls_back_to_template():
    ev = fixtures.grounding()
    client = _ScriptedLLMClient([])  # raises immediately
    text = generate_explanation("find buildings", ev, ev.confidence, llm_client=client)
    ctx = build_grounding_context(ev)
    assert check_numeric_claims(text, ctx).ok


def test_malformed_format_string_retries_with_guidance_instead_of_giving_up_immediately():
    # Regression: a format-spec mistake like {token:.1f} used to be caught
    # by the same broad except as "client unavailable" and break the loop
    # immediately -- wasting the one retry attempt on a fixable mistake
    # instead of asking again. Confirm it now retries and can still recover.
    ev = fixtures.change_detection()
    client = _ScriptedLLMClient([
        "About {changed_area_pct:.1f}% of the area changed.",  # bad: format spec on a str token
        "About {changed_area_pct}% of the area changed, fixed.",
    ])
    text = generate_explanation("what changed", ev, ev.confidence, llm_client=client)
    ctx = build_grounding_context(ev)
    assert check_numeric_claims(text, ctx).ok
    assert len(client.calls) == 2  # it got to use its retry, not just give up
    assert "fixed" in text


def test_stray_brace_falls_back_to_template_after_using_the_retry_budget():
    ev = fixtures.change_detection()
    client = _ScriptedLLMClient([
        "score was 0.5}",   # unmatched brace on both attempts
        "still broken {",
    ])
    text = generate_explanation("what changed", ev, ev.confidence, llm_client=client)
    ctx = build_grounding_context(ev)
    assert check_numeric_claims(text, ctx).ok  # safe template, no crash
    assert len(client.calls) == 2
