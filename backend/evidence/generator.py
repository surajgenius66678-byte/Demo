"""
Explanation generation — evidence + query -> answer_text.

Flow:
  1. Build a GroundingContext (the only numbers allowed to appear).
  2. If no llm_client was given, skip straight to the deterministic
     template (templates.py) — the default, zero-setup path.
  3. Otherwise, prompt the LLM with the query, the confidence band (so it
     can hedge language appropriately), and the exact {token} placeholders
     it's allowed to use for any number — explicitly told never to type a
     digit itself.
  4. Fill placeholders via str.format_map with the grounding dict.
  5. Run the numeric claim-check (claim_check.py) on the filled text.
  6. On failure: retry with a stricter prompt naming exactly which number(s)
     leaked, up to config.MAX_LLM_REGENERATION_ATTEMPTS times. If every
     attempt fails (or the client errors at any point): fall back to
     templates.py, which is guaranteed to pass the claim-check by
     construction.

Every exit path is claim-check-clean before it becomes answer_text —
service.py re-checks the return value again anyway, as a final safety net.
"""
from __future__ import annotations

from backend.evidence import templates
from backend.evidence.claim_check import check_numeric_claims
from backend.evidence.config import MAX_LLM_REGENERATION_ATTEMPTS
from backend.evidence.grounding import GroundingContext, build_grounding_context
from backend.evidence.llm_client import ExplanationLLMClient
from backend.shared.schemas import Confidence, Evidence


class _SafeFormatDict(dict):
    """format_map() that leaves an unknown {placeholder} as literal text
    instead of raising KeyError — an LLM inventing a token name should
    degrade to visibly-odd text, never crash the response."""

    def __missing__(self, key):
        return "{" + str(key) + "}"


def generate_explanation(
    query: str,
    evidence: Evidence,
    confidence: Confidence,
    llm_client: ExplanationLLMClient | None = None,
) -> str:
    ctx = build_grounding_context(evidence)

    if llm_client is None:
        return templates.render_template(evidence, ctx)

    feedback = None
    for _ in range(MAX_LLM_REGENERATION_ATTEMPTS + 1):
        try:
            raw = llm_client.generate(_system_prompt(), _user_prompt(query, evidence, confidence, ctx, feedback))
        except Exception:
            # The client itself failed (network, auth, timeout, ...) — that
            # won't fix itself by asking the same client again immediately,
            # so stop and fall through to the guaranteed-safe template.
            break

        try:
            filled = raw.format_map(_SafeFormatDict(ctx.as_format_dict()))
        except Exception as exc:
            # The LLM's own text wasn't a valid format string — a stray
            # brace, a format spec like {token:.1f}, or a positional field
            # like {0}/{}. That's a fixable mistake, not client
            # unavailability, so spend a retry on specific guidance instead
            # of giving up immediately and wasting the retry budget.
            feedback = (
                f"Your previous answer wasn't usable ({exc}). Use plain "
                "prose with bare {token} placeholders only — no format "
                "specs like {token:.1f}, no positional fields like {0} or "
                "{}, and make sure every '{' has a matching '}'."
            )
            continue

        result = check_numeric_claims(filled, ctx)
        if result.ok:
            return filled.strip() + templates.warnings_suffix(evidence)

        feedback = (
            f"Your previous answer contained number(s) not present in the "
            f"evidence: {result.unverified_numbers}. Use ONLY the {{token}} "
            "placeholders listed above for any number — do not type digits "
            "yourself."
        )

    return templates.render_template(evidence, ctx)


def _system_prompt() -> str:
    return (
        "You write one short, plain-language paragraph answering a "
        "question about satellite imagery analysis, for a non-technical "
        "reader. You are NOT allowed to write any digit yourself. Every "
        "number must be one of the exact {token} placeholders you're "
        "given — copy them verbatim, including the curly braces. Do not "
        "invent a percentage, count, or score that isn't offered to you as "
        "a token. Do not mention specific model or software names."
    )


def _user_prompt(
    query: str,
    evidence: Evidence,
    confidence: Confidence,
    ctx: GroundingContext,
    feedback: str | None,
) -> str:
    lines = [
        f"User's question: {query}",
        f"Analysis task: {evidence.task.value}",
        f"Confidence level: {confidence.band} — hedge language accordingly "
        "(avoid definitive claims at LOW confidence).",
        "Available number placeholders (use only these for any digit):",
    ]
    for token, value in ctx.tokens.items():
        lines.append(f"  {{{token}}} = {value}")
    if evidence.vqa_answer_raw:
        lines.append(f"Raw model description to paraphrase: {evidence.vqa_answer_raw}")
    if feedback:
        lines.append(feedback)
    lines.append("Write the answer now, as prose, one paragraph.")
    return "\n".join(lines)
