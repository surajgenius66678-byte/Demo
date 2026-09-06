"""
Deterministic, non-LLM sentence templates — the guaranteed-safe fallback.

Every {token} used here comes from grounding.GroundingContext, so by
construction nothing rendered here can contain a number that isn't already
in Evidence. This is also the final safety net generator.py reaches for if
the LLM path's numeric claim-check keeps failing.
"""
from __future__ import annotations

from backend.evidence.grounding import GroundingContext
from backend.shared.schemas import Evidence, TaskType

_TEMPLATES: dict[TaskType, str] = {
    TaskType.SINGLE_IMAGE_VQA:
        "{vqa_answer}",
    TaskType.CAPTIONING:
        "{vqa_answer}",
    TaskType.GROUNDING:
        "Found {detection_count_phrase} for '{detection_labels}', with "
        "detection confidence {detection_confidence_phrase}.",
    TaskType.CHANGE_DETECTION:
        "{changed_area_pct}% of the analyzed area ({changed_area_px} pixels) "
        "shows change between the two images, at a mean confidence of "
        "{change_mean_confidence}%.",
    TaskType.CHANGE_VQA:
        "{vqa_answer} ({changed_area_pct}% of the area changed, mean "
        "confidence {change_mean_confidence}%.)",
    TaskType.OPTICAL_SAR_FUSION:
        "Combining optical and SAR evidence: {vqa_answer}",
}


def render_template(evidence: Evidence, ctx: GroundingContext) -> str:
    template = _TEMPLATES.get(evidence.task)
    if template is None:
        return _generic_fallback(evidence, ctx) + warnings_suffix(evidence)

    format_dict = dict(ctx.as_format_dict())
    format_dict.setdefault(
        "vqa_answer",
        (evidence.vqa_answer_raw or "").strip() or "No description was returned.",
    )

    try:
        text = template.format(**format_dict)
    except KeyError:
        # This task's template expects a field this particular Evidence
        # didn't carry (e.g. CHANGE_DETECTION with no change_map at all) —
        # degrade to whatever raw description exists rather than crash.
        text = _generic_fallback(evidence, ctx)

    return text + warnings_suffix(evidence)


def _generic_fallback(evidence: Evidence, ctx: GroundingContext) -> str:
    if evidence.vqa_answer_raw:
        return evidence.vqa_answer_raw.strip()
    if ctx.tokens:
        pairs = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in ctx.tokens.items())
        return f"Evidence summary — {pairs}."
    return "The analysis completed but produced no describable evidence."


def warnings_suffix(evidence: Evidence) -> str:
    """Non-coregistration warnings get surfaced as a trailing caveat.
    (A coregistration warning always triggers an abstain in validator.py
    before this module is ever reached, but we filter defensively anyway
    in case that upstream rule ever changes.)
    """
    non_coreg = [w for w in (evidence.warnings or []) if "coregistration" not in w.lower()]
    if not non_coreg:
        return ""
    return " Note: " + "; ".join(non_coreg) + "."
