"""
Part 5's public interface — the ONE function Part 2 calls:

    validate_and_respond(query: str, evidence: Evidence) -> FinalResponse

Everything else in this package is an implementation detail behind this
function. `llm_client` is an additive, optional third parameter — Part 2's
contract call `validate_and_respond(query, evidence)` still works unchanged;
pass a client (see llm_client.py) once a local model is wired up.
"""
from __future__ import annotations

from backend.evidence import templates
from backend.evidence.claim_check import check_numeric_claims
from backend.evidence.confidence import compute_confidence
from backend.evidence.generator import generate_explanation
from backend.evidence.grounding import build_grounding_context
from backend.evidence.llm_client import ExplanationLLMClient
from backend.evidence.validator import validate_evidence
from backend.shared.schemas import Confidence, Evidence, FinalResponse


def validate_and_respond(
    query: str,
    evidence: Evidence,
    llm_client: ExplanationLLMClient | None = None,
) -> FinalResponse:
    validation = validate_evidence(query, evidence)
    if validation.should_abstain:
        reason = validation.reason or "This question can't be answered from the available evidence."
        return FinalResponse(
            answer_text=reason,
            confidence=Confidence(value=None, band="LOW", basis=reason),
            evidence=evidence,
            abstained=True,
            abstain_reason=validation.reason,
        )

    confidence = compute_confidence(evidence)
    answer_text = generate_explanation(query, evidence, confidence, llm_client)

    # Final safety net, at the outermost boundary: re-verify every number in
    # answer_text traces to evidence, no matter which internal path produced
    # it. generate_explanation already guarantees a claim-check-clean
    # result, so this should never actually trip — but a response that
    # could contain an untraceable number must never reach the user, so we
    # check again here rather than trust the caller.
    ctx = build_grounding_context(evidence)
    if not check_numeric_claims(answer_text, ctx).ok:
        answer_text = templates.render_template(evidence, ctx)

    return FinalResponse(
        answer_text=answer_text,
        confidence=confidence,
        evidence=evidence,
        abstained=False,
        abstain_reason=None,
    )
