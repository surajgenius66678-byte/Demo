"""
Hand-written Evidence fixtures — one per TaskType, plus edge cases, used
across the whole test suite. Building these by hand (not via Part 4) is
exactly the "Mocking strategy: none needed inward" note in the architecture
doc — Part 5 is fully testable against these alone, no GPU or model needed.
"""
from backend.shared.schemas import (
    ChangeMap,
    Confidence,
    Detection,
    Evidence,
    Modality,
    TaskType,
)


def single_image_vqa() -> Evidence:
    return Evidence(
        task=TaskType.SINGLE_IMAGE_VQA,
        model_used="vlm-remote-sensing-v1",
        modality_used=[Modality.OPTICAL],
        vqa_answer_raw="The image shows an agricultural field bordered by a river.",
        confidence=Confidence(value=0.81, band="HIGH", basis="VLM softmax confidence"),
    )


def captioning() -> Evidence:
    return Evidence(
        task=TaskType.CAPTIONING,
        model_used="vlm-remote-sensing-v1",
        modality_used=[Modality.OPTICAL],
        vqa_answer_raw="A dense urban area with a grid street pattern near a coastline.",
        confidence=Confidence(value=0.72, band="MEDIUM", basis="VLM softmax confidence"),
    )


def grounding() -> Evidence:
    return Evidence(
        task=TaskType.GROUNDING,
        model_used="grounding-dino-rs",
        modality_used=[Modality.OPTICAL],
        detections=[
            Detection(label="building", box_px=[10, 10, 40, 40], mask_rle=None, score=0.91),
            Detection(label="building", box_px=[100, 80, 160, 130], mask_rle=None, score=0.77),
            Detection(label="building", box_px=[200, 200, 240, 260], mask_rle=None, score=0.64),
        ],
        confidence=Confidence(value=None, band="MEDIUM", basis="not yet aggregated upstream"),
    )


def change_detection() -> Evidence:
    return Evidence(
        task=TaskType.CHANGE_DETECTION,
        model_used="change-former-rs",
        modality_used=[Modality.OPTICAL],
        change_map=ChangeMap(
            probability_raster_path="/data/jobs/abc/change_prob.tif",
            changed_area_px=48213,
            changed_area_pct=12.4,
            mean_confidence=0.68,
        ),
        confidence=Confidence(value=0.68, band="MEDIUM", basis="change-map mean confidence"),
    )


def change_vqa() -> Evidence:
    return Evidence(
        task=TaskType.CHANGE_VQA,
        model_used="change-former-rs+vlm-remote-sensing-v1",
        modality_used=[Modality.OPTICAL],
        vqa_answer_raw="New construction appears in the northeast quadrant.",
        change_map=ChangeMap(
            probability_raster_path="/data/jobs/def/change_prob.tif",
            changed_area_px=15320,
            changed_area_pct=4.1,
            mean_confidence=0.74,
        ),
        confidence=Confidence(value=0.74, band="MEDIUM", basis="change-map mean confidence"),
    )


def optical_sar_fusion() -> Evidence:
    return Evidence(
        task=TaskType.OPTICAL_SAR_FUSION,
        model_used="fusion-net-rs",
        modality_used=[Modality.OPTICAL, Modality.SAR],
        vqa_answer_raw="Flooding is visible in low-lying areas, confirmed by both sensors.",
        stats={"flooded_area_km2": 3.2, "sar_backscatter_drop_db": 6.5},
        confidence=Confidence(value=0.79, band="HIGH", basis="cross-modality agreement score"),
    )


def unsupported() -> Evidence:
    return Evidence(
        task=TaskType.UNSUPPORTED,
        model_used="none",
        modality_used=[],
        confidence=Confidence(value=None, band="LOW", basis="task not supported"),
    )


def weak_empty() -> Evidence:
    """Deliberately empty — nothing came back. Must trigger abstain."""
    return Evidence(
        task=TaskType.SINGLE_IMAGE_VQA,
        model_used="vlm-remote-sensing-v1",
        modality_used=[Modality.OPTICAL],
        confidence=Confidence(value=None, band="LOW", basis="no signal returned"),
    )


def weak_low_confidence() -> Evidence:
    """Not empty, but every real signal sits below the weak-evidence floor."""
    return Evidence(
        task=TaskType.GROUNDING,
        model_used="grounding-dino-rs",
        modality_used=[Modality.OPTICAL],
        detections=[Detection(label="building", box_px=[0, 0, 5, 5], mask_rle=None, score=0.08)],
        confidence=Confidence(value=0.11, band="LOW", basis="single low-score detection"),
    )


def coregistration_refused() -> Evidence:
    return Evidence(
        task=TaskType.CHANGE_DETECTION,
        model_used="change-former-rs",
        modality_used=[Modality.OPTICAL],
        confidence=Confidence(value=None, band="LOW", basis="task refused before inference"),
        warnings=["coregistration failed: offset 42.0px exceeds threshold"],
    )


ALL_TASK_FIXTURES = {
    TaskType.SINGLE_IMAGE_VQA: single_image_vqa,
    TaskType.CAPTIONING: captioning,
    TaskType.GROUNDING: grounding,
    TaskType.CHANGE_DETECTION: change_detection,
    TaskType.CHANGE_VQA: change_vqa,
    TaskType.OPTICAL_SAR_FUSION: optical_sar_fusion,
    TaskType.UNSUPPORTED: unsupported,
}
