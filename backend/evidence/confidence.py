"""
Confidence scoring — Part 5 owns the confidence on the *response*, which
isn't necessarily identical to the object Part 4 attached to Evidence.

Rule from the architecture doc: "confidence.value always traces to a field
already in evidence.confidence or a deterministic function of
evidence.detections[*].score / evidence.change_map.mean_confidence — never
a bare LLM-stated number." Nothing in this file calls an LLM.
"""
from __future__ import annotations

from backend.evidence.config import CONFIDENCE_HIGH_THRESHOLD, CONFIDENCE_MEDIUM_THRESHOLD
from backend.shared.schemas import Confidence, Evidence


def compute_confidence(evidence: Evidence) -> Confidence:
    value, basis = _derive_value(evidence)
    return Confidence(value=value, band=_band_for(value), basis=basis)


def _derive_value(evidence: Evidence) -> tuple[float | None, str]:
    signals: list[tuple[float, str]] = []

    upstream = evidence.confidence
    if upstream is not None and upstream.value is not None:
        signals.append((upstream.value, f"model-reported confidence ({upstream.basis})"))

    if evidence.detections:
        scores = [d.score for d in evidence.detections]
        mean_score = sum(scores) / len(scores)
        signals.append((mean_score, f"mean of {len(scores)} detection score(s)"))

    if evidence.change_map is not None:
        signals.append((evidence.change_map.mean_confidence, "change-map mean confidence"))

    if not signals:
        return None, "no calibrated confidence signal was present in evidence"

    # Conservative by design: take the lowest of the available real signals,
    # not the highest or an average, so response confidence never overstates
    # the weakest link in the evidence chain.
    value, basis = min(signals, key=lambda s: s[0])
    return round(value, 4), basis


def _band_for(value: float | None) -> str:
    if value is None:
        return "LOW"
    if value >= CONFIDENCE_HIGH_THRESHOLD:
        return "HIGH"
    if value >= CONFIDENCE_MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"
