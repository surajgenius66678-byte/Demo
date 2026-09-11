from __future__ import annotations

from shared.schemas import Evidence


class EvidenceAggregator:
    """
    Combines evidence produced by multiple specialist tasks.

    The aggregator does not invent new evidence.
    It only combines information that already exists in the
    specialist Evidence objects.
    """

    def aggregate(self, evidences: list[Evidence]) -> Evidence:
        if not evidences:
            raise ValueError("Cannot aggregate an empty evidence list")

        primary = evidences[0]

        # Preserve detections from every specialist.
        detections = []
        for evidence in evidences:
            detections.extend(evidence.detections)

        # Preserve all warnings.
        warnings = []
        for evidence in evidences:
            warnings.extend(evidence.warnings)

        # Preserve useful statistics.
        stats: dict[str, float] = {}
        for evidence in evidences:
            stats.update(evidence.stats)

        # Prefer an available change map.
        change_map = primary.change_map
        if change_map is None:
            for evidence in evidences:
                if evidence.change_map is not None:
                    change_map = evidence.change_map
                    break

        # Preserve VQA answers from specialist outputs.
        answers = []
        for evidence in evidences:
            if evidence.vqa_answer_raw:
                answers.append(evidence.vqa_answer_raw)

        combined_answer = None
        if answers:
            combined_answer = "\n".join(answers)

        # Use the lowest confidence among contributing evidence.
        # This is intentionally conservative.
        confidence_values = [
            evidence.confidence.value
            for evidence in evidences
            if evidence.confidence.value is not None
        ]

        if confidence_values:
            confidence_value = min(confidence_values)
        else:
            confidence_value = None

        if confidence_value is None:
            confidence_band = primary.confidence.band
        elif confidence_value >= 0.80:
            confidence_band = "HIGH"
        elif confidence_value >= 0.50:
            confidence_band = "MEDIUM"
        else:
            confidence_band = "LOW"

        contributing_models = []
        for evidence in evidences:
            if evidence.model_used not in contributing_models:
                contributing_models.append(evidence.model_used)

        contributing_modalities = []
        for evidence in evidences:
            for modality in evidence.modality_used:
                if modality not in contributing_modalities:
                    contributing_modalities.append(modality)

        return Evidence(
            task=primary.task,
            model_used=", ".join(contributing_models),
            modality_used=contributing_modalities,
            detections=detections,
            change_map=change_map,
            vqa_answer_raw=combined_answer,
            stats=stats,
            confidence=primary.confidence.model_copy(
                update={
                    "value": confidence_value,
                    "band": confidence_band,
                    "basis": "conservative aggregation of specialist evidence",
                }
            ),
            warnings=warnings,
        )