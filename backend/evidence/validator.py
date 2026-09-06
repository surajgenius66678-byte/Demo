"""
Evidence validator — Part 5's hallucination gate.

Decides whether the Evidence Part 4 produced is actually enough to answer
the user's query, or whether the honest response is to abstain. This runs
*before* any LLM call: an abstain never touches the explanation generator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from backend.evidence.config import COREGISTRATION_WARNING_MARKER, WEAK_EVIDENCE_FLOOR
from backend.shared.schemas import Evidence, TaskType


@dataclass
class ValidationResult:
    should_abstain: bool
    reason: Optional[str] = None


def validate_evidence(query: str, evidence: Evidence) -> ValidationResult:
    """Checked from hardest "no" to softest "no" — the first match wins."""
    if evidence is None:
        return ValidationResult(True, "No evidence was returned for this query.")

    if evidence.task == TaskType.UNSUPPORTED:
        return ValidationResult(
            True,
            "This question isn't something the system currently supports — "
            "it falls outside the set of analyses SatQuery AI can run.",
        )

    coreg_warning = _first_matching_warning(evidence.warnings, COREGISTRATION_WARNING_MARKER)
    if coreg_warning:
        return ValidationResult(
            True,
            "The two images couldn't be reliably aligned, so a change-based "
            f"answer would not be trustworthy ({coreg_warning}).",
        )

    if _is_empty_evidence(evidence):
        return ValidationResult(
            True,
            "No usable signal came back from the analysis for this query.",
        )

    if _is_weak_evidence(evidence):
        return ValidationResult(
            True,
            "The available evidence is too weak to support a confident answer.",
        )

    return ValidationResult(False, None)


def _first_matching_warning(warnings: list[str], marker: str) -> Optional[str]:
    marker = marker.lower()
    for w in warnings or []:
        if marker in w.lower():
            return w
    return None


def _is_empty_evidence(evidence: Evidence) -> bool:
    """True only when literally nothing came back — every optional signal
    is unset. A legitimate "zero buildings found" (an empty detections list
    alongside a populated vqa_answer_raw or stats) is NOT empty evidence.
    """
    has_detections = bool(evidence.detections)
    has_change_map = evidence.change_map is not None
    has_vqa = bool(evidence.vqa_answer_raw and evidence.vqa_answer_raw.strip())
    has_stats = bool(evidence.stats)
    return not (has_detections or has_change_map or has_vqa or has_stats)


def _is_weak_evidence(evidence: Evidence) -> bool:
    """True when a calibrated confidence value exists but sits below the
    floor, and there's no independent stronger signal to lean on instead.
    """
    conf = evidence.confidence
    if conf is None or conf.value is None:
        return False  # nothing calibrated to judge weakness by — not this function's call
    if conf.value >= WEAK_EVIDENCE_FLOOR:
        return False
    if evidence.detections and max(d.score for d in evidence.detections) >= WEAK_EVIDENCE_FLOOR:
        return False
    if evidence.change_map is not None and evidence.change_map.mean_confidence >= WEAK_EVIDENCE_FLOOR:
        return False
    return True
