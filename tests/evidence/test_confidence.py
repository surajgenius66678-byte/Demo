from backend.evidence.confidence import compute_confidence
from backend.shared.schemas import Confidence, Detection, Evidence, Modality, TaskType
from tests.evidence import fixtures


def test_uses_upstream_confidence_when_present():
    conf = compute_confidence(fixtures.change_detection())
    assert conf.value is not None
    assert conf.band in ("LOW", "MEDIUM", "HIGH")


def test_derives_from_detection_scores_when_no_upstream_value():
    ev = fixtures.grounding()  # confidence.value is None in this fixture
    conf = compute_confidence(ev)
    scores = [d.score for d in ev.detections]
    assert conf.value == round(sum(scores) / len(scores), 4)


def test_missing_all_signals_gives_low_band_and_none_value():
    conf = compute_confidence(fixtures.weak_empty())
    assert conf.value is None
    assert conf.band == "LOW"


def test_conservative_take_the_minimum_signal():
    ev = Evidence(
        task=TaskType.GROUNDING,
        model_used="m",
        modality_used=[Modality.OPTICAL],
        detections=[Detection(label="x", box_px=[0, 0, 1, 1], mask_rle=None, score=0.2)],
        confidence=Confidence(value=0.9, band="HIGH", basis="upstream"),
    )
    conf = compute_confidence(ev)
    # upstream says 0.9, detections say 0.2 -> the conservative pick is 0.2
    assert conf.value == 0.2


def test_band_thresholds():
    for value, expected_band in [(0.9, "HIGH"), (0.75, "HIGH"), (0.5, "MEDIUM"), (0.4, "MEDIUM"), (0.1, "LOW")]:
        ev = Evidence(
            task=TaskType.SINGLE_IMAGE_VQA,
            model_used="m",
            modality_used=[Modality.OPTICAL],
            vqa_answer_raw="x",
            confidence=Confidence(value=value, band="?", basis="test"),
        )
        assert compute_confidence(ev).band == expected_band
