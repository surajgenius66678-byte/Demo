"""
Post-generation numeric claim check.

Extracts every number-looking token from generated prose and confirms each
one traces back to a real value in Evidence (via GroundingContext's
allow-set). This is the backstop for the "numbers must be templated in, not
freely generated" rule — it catches an LLM that ignored the placeholder
instruction and typed a digit itself.

Known trade-off: number extraction is a regex heuristic (see
grounding.extract_numbers), so an alphanumeric identifier like "Sentinel-2"
could in principle be misread as the number 2 and, if 2 isn't otherwise
grounded, trip a false-positive fallback to the templated sentence. That
failure mode trades a slightly duller answer for never risking an
ungrounded number reaching the user — an acceptable direction to fail in an
evidence-first system. The generator's prompt also asks the LLM not to
mention model/product names in prose, which avoids this in practice.
"""
from __future__ import annotations

from dataclasses import dataclass

from backend.evidence.config import CLAIM_CHECK_DECIMAL_PLACES, CLAIM_CHECK_RELATIVE_TOLERANCE
from backend.evidence.grounding import GroundingContext, extract_numbers


@dataclass
class ClaimCheckResult:
    ok: bool
    unverified_numbers: list[float]


def check_numeric_claims(text: str, ctx: GroundingContext) -> ClaimCheckResult:
    found = extract_numbers(text)
    unverified = [n for n in found if not _is_grounded(n, ctx.allowed_values)]
    return ClaimCheckResult(ok=not unverified, unverified_numbers=unverified)


def _is_grounded(value: float, allowed: set[float]) -> bool:
    for a in allowed:
        if round(value, CLAIM_CHECK_DECIMAL_PLACES) == round(a, CLAIM_CHECK_DECIMAL_PLACES):
            return True
        denom = max(abs(a), 1e-9)
        if abs(value - a) / denom <= CLAIM_CHECK_RELATIVE_TOLERANCE:
            return True
    return False
