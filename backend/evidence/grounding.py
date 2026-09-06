"""
Turns an Evidence object into:

  1. a `GroundingContext.tokens` dict — named, pre-formatted numeric strings
     the explanation generator is allowed to drop into prose via a
     `{token}` placeholder, and
  2. a `GroundingContext.allowed_values` set — every real numeric value
     found anywhere in evidence (including numbers the upstream model
     already stated in `vqa_answer_raw`), used by claim_check.py to verify
     that nothing in the final answer_text is untraceable to evidence.

This is the one place that reads every numeric field out of Evidence, so
the "a number in the response must come from here" rule in generator.py has
a single, auditable source instead of being scattered across the package.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.shared.schemas import Evidence

#
# The leading `(?<![\w.])` is a negative lookbehind: a `-` only counts as a
# minus sign if it's NOT immediately glued to a preceding digit/letter/dot.
# Without it, a hyphenated range like "64-91%" parses as [64.0, -91.0] (the
# second number picks up the range's hyphen as a sign) and a date like
# "2024-03-15" parses as [2024.0, -3.0, -15.0] — both false "negative
# number" reads that can make ordinary, correct text fail the claim-check.
# A genuine negative number ("-12.5 meters", "(-12.5)") still matches fine,
# since a space/paren/start-of-string isn't a word char or a dot.
_NUMBER_RE = re.compile(r"(?<![\w.])-?\d[\d,]*\.?\d*")


def extract_numbers(text: str) -> list[float]:
    """Pull every number-looking substring out of free text as a float.

    Strips thousands-separator commas first (`"1,234"` -> `1234.0`). This is
    a regex heuristic, not a full parser — see claim_check.py's module
    docstring for the known trade-off around alphanumeric identifiers.
    """
    out = []
    for match in _NUMBER_RE.findall(text or ""):
        cleaned = match.replace(",", "")
        if cleaned in ("", "-", "."):
            continue
        try:
            out.append(float(cleaned))
        except ValueError:
            continue
    return out


def _fmt(value: float, decimals: int = 2) -> str:
    """Format a float for display, trimming a trailing `.00` but never
    touching digits that belong to the integer part (a naive
    `.rstrip('0')` on a no-decimal-point string like "48210" would
    corrupt it into "4821" — guard on "." being present first).
    """
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text if text else "0"


@dataclass
class GroundingContext:
    # token name -> already-formatted display string, safe to drop into prose
    tokens: dict[str, str] = field(default_factory=dict)
    # every real numeric value found anywhere in evidence, for claim-check
    allowed_values: set[float] = field(default_factory=set)

    def as_format_dict(self) -> dict[str, str]:
        return dict(self.tokens)


def build_grounding_context(evidence: Evidence) -> GroundingContext:
    ctx = GroundingContext()

    # --- stats: dict[str, float] -> one token per key, verbatim ------------
    for key, value in (evidence.stats or {}).items():
        token = _dedupe_token(_safe_token(key), ctx.tokens)
        ctx.tokens[token] = _fmt(value)
        ctx.allowed_values.add(round(float(value), 6))

    # --- change_map ----------------------------------------------------------
    if evidence.change_map is not None:
        cm = evidence.change_map
        ctx.tokens["changed_area_px"] = _fmt(cm.changed_area_px, 0)
        ctx.tokens["changed_area_pct"] = _fmt(cm.changed_area_pct)
        ctx.tokens["change_mean_confidence"] = _fmt(cm.mean_confidence * 100)
        ctx.allowed_values.update({
            round(float(cm.changed_area_px), 6),
            round(float(cm.changed_area_pct), 6),
            round(float(cm.mean_confidence), 6),
            round(float(cm.mean_confidence) * 100, 6),
        })

    # --- detections ------------------------------------------------------------
    if evidence.detections:
        scores = [d.score for d in evidence.detections]
        mean_score = sum(scores) / len(scores)
        count = len(evidence.detections)

        ctx.tokens["detection_count"] = str(count)
        ctx.tokens["detection_min_score"] = _fmt(min(scores) * 100)
        ctx.tokens["detection_max_score"] = _fmt(max(scores) * 100)
        ctx.tokens["detection_mean_score"] = _fmt(mean_score * 100)
        ctx.tokens["detection_labels"] = ", ".join(sorted({d.label for d in evidence.detections}))
        # Pre-phrased, grammatically-correct fragments for the deterministic
        # template (templates.py stays plain string substitution — no
        # singular/plural or single-value-range branching lives there).
        ctx.tokens["detection_count_phrase"] = f"{count} match" + ("" if count == 1 else "es")
        ctx.tokens["detection_confidence_phrase"] = (
            f"around {_fmt(min(scores) * 100)}%"
            if min(scores) == max(scores)
            else f"ranging from {_fmt(min(scores) * 100)}% to {_fmt(max(scores) * 100)}%"
        )

        ctx.allowed_values.add(float(count))
        # The mean is a real, deterministically-derived number too — without
        # adding it here, a response that (correctly) uses
        # {detection_mean_score} could fail the claim-check just because the
        # average itself isn't within tolerance of any single score.
        ctx.allowed_values.add(round(mean_score, 6))
        ctx.allowed_values.add(round(mean_score * 100, 6))
        for s in scores:
            ctx.allowed_values.add(round(float(s), 6))
            ctx.allowed_values.add(round(float(s) * 100, 6))

    # --- confidence ----------------------------------------------------------
    if evidence.confidence is not None and evidence.confidence.value is not None:
        ctx.tokens["confidence_pct"] = _fmt(evidence.confidence.value * 100)
        ctx.allowed_values.add(round(float(evidence.confidence.value), 6))
        ctx.allowed_values.add(round(float(evidence.confidence.value) * 100, 6))

    # --- vqa_answer_raw: numbers the upstream model already stated ---------
    # These are already "in the evidence dict" (vqa_answer_raw is a real
    # Evidence field), so a paraphrase repeating one isn't a hallucination.
    # They only widen the claim-check allow-set — never become {tokens},
    # since raw model text isn't guaranteed well-formed enough to template.
    if evidence.vqa_answer_raw:
        for n in extract_numbers(evidence.vqa_answer_raw):
            ctx.allowed_values.add(round(n, 6))

    # --- warnings: same reasoning as vqa_answer_raw above -------------------
    # A warning is real Evidence content too (e.g. "batch size reduced from
    # 32 to 8" from a retry-on-OOM path). Without this, a non-coregistration
    # warning appended to the answer (see templates.warnings_suffix) could
    # make service.py's final claim-check reject an otherwise-good answer
    # purely because of numbers inside the warning text it never scanned.
    for w in evidence.warnings or []:
        for n in extract_numbers(w):
            ctx.allowed_values.add(round(n, 6))

    return ctx


def _safe_token(key: str) -> str:
    """stats keys become {token} names — keep them str.format()-safe."""
    return re.sub(r"\W+", "_", key.strip()).strip("_") or "stat"


def _dedupe_token(token: str, existing: dict[str, str]) -> str:
    """Two different stats keys can sanitize to the same token name (e.g.
    "flood-area-km2" and "flood_area_km2" both become "flood_area_km2") —
    without this, the second would silently overwrite the first in `tokens`
    and make it unreachable as a placeholder. Suffix instead of clobbering.
    """
    if token not in existing:
        return token
    suffix = 2
    while f"{token}_{suffix}" in existing:
        suffix += 1
    return f"{token}_{suffix}"
