from backend.evidence.claim_check import check_numeric_claims
from backend.evidence.grounding import build_grounding_context
from tests.evidence import fixtures


def test_grounded_number_passes():
    ev = fixtures.change_detection()
    ctx = build_grounding_context(ev)
    text = f"About {ctx.tokens['changed_area_pct']}% of the area changed."
    assert check_numeric_claims(text, ctx).ok


def test_hallucinated_number_fails():
    ev = fixtures.change_detection()  # changed_area_pct = 12.4
    ctx = build_grounding_context(ev)
    result = check_numeric_claims("About 87% of the area changed.", ctx)
    assert not result.ok
    assert 87.0 in result.unverified_numbers


def test_naturally_rounded_number_passes_within_tolerance():
    ev = fixtures.change_detection()  # changed_area_pct = 12.4
    ctx = build_grounding_context(ev)
    # "about 12%" for a true value of 12.4 is ordinary rounding, not fabrication
    result = check_numeric_claims("About 12% of the area changed.", ctx)
    assert result.ok


def test_comma_formatted_number_passes():
    ev = fixtures.change_detection()  # changed_area_px = 48213
    ctx = build_grounding_context(ev)
    result = check_numeric_claims("Roughly 48,213 pixels changed.", ctx)
    assert result.ok


def test_number_stated_in_vqa_answer_raw_is_grounded():
    # A number the upstream model already stated in vqa_answer_raw is
    # already "in the evidence dict" — repeating it isn't a hallucination,
    # even though it never becomes a {token} (see grounding.py).
    from backend.shared.schemas import Confidence, Evidence, Modality, TaskType

    ev = Evidence(
        task=TaskType.SINGLE_IMAGE_VQA,
        model_used="vlm-remote-sensing-v1",
        modality_used=[Modality.OPTICAL],
        vqa_answer_raw="The field covers approximately 3.2 square kilometers.",
        confidence=Confidence(value=0.8, band="HIGH", basis="test"),
    )
    ctx = build_grounding_context(ev)
    result = check_numeric_claims("The field covers about 3.2 square kilometers.", ctx)
    assert result.ok


def test_multiple_numbers_one_bad_fails_and_reports_only_the_bad_one():
    ev = fixtures.change_detection()  # changed_area_pct = 12.4
    ctx = build_grounding_context(ev)
    result = check_numeric_claims("12.4% changed, confidence was 999%.", ctx)
    assert not result.ok
    assert 999.0 in result.unverified_numbers
    assert 12.4 not in result.unverified_numbers


# --- regression tests for bugs found in review -------------------------------

def test_hyphenated_range_is_not_misread_as_a_negative_number():
    # Regression: "64-91%" used to extract as [64.0, -91.0] -- the range's
    # hyphen got read as a minus sign on 91, so an ordinary "X-Y%" sentence
    # would fail the claim-check even with both endpoints genuinely grounded.
    ev = fixtures.grounding()  # detection scores 0.91, 0.77, 0.64 -> 91%, 77%, 64%
    ctx = build_grounding_context(ev)
    result = check_numeric_claims("confidence 64-91%", ctx)
    assert result.ok, result.unverified_numbers


def test_genuine_negative_number_still_extracted_correctly():
    # The lookbehind fix must not break real negative numbers.
    from backend.evidence.grounding import extract_numbers

    assert extract_numbers("elevation dropped by -12.5 meters") == [-12.5]
    assert extract_numbers("(-3.0)") == [-3.0]


def test_detection_mean_score_token_is_itself_grounded():
    # Regression: detection_mean_score was computed and offered as a
    # {token}, but the mean value itself was never added to allowed_values
    # -- only the individual scores were. A response correctly using the
    # mean could fail claim-check whenever the average wasn't coincidentally
    # close to one specific score.
    from backend.shared.schemas import Confidence, Detection, Evidence, Modality, TaskType

    ev = Evidence(
        task=TaskType.GROUNDING,
        model_used="m",
        modality_used=[Modality.OPTICAL],
        detections=[
            Detection(label="building", box_px=[0, 0, 1, 1], mask_rle=None, score=0.91),
            Detection(label="building", box_px=[0, 0, 1, 1], mask_rle=None, score=0.50),
            Detection(label="building", box_px=[0, 0, 1, 1], mask_rle=None, score=0.30),
        ],
        confidence=Confidence(value=None, band="LOW", basis="x"),
    )
    ctx = build_grounding_context(ev)
    assert ctx.tokens["detection_mean_score"] == "57"
    result = check_numeric_claims("Mean detection confidence was 57%.", ctx)
    assert result.ok, result.unverified_numbers


def test_warning_numbers_are_grounded():
    # Regression: a non-coregistration warning's own numbers (e.g. "batch
    # size reduced from 32 to 8") were never scanned into allowed_values, so
    # appending that warning via templates.warnings_suffix could make
    # service.py's final safety-net check reject an otherwise-good answer.
    from backend.shared.schemas import ChangeMap, Confidence, Evidence, Modality, TaskType

    ev = Evidence(
        task=TaskType.CHANGE_DETECTION,
        model_used="m",
        modality_used=[Modality.OPTICAL],
        change_map=ChangeMap(probability_raster_path="p", changed_area_px=1000, changed_area_pct=10.0, mean_confidence=0.6),
        confidence=Confidence(value=0.6, band="MEDIUM", basis="x"),
        warnings=["retried after OOM: batch size reduced from 32 to 8"],
    )
    ctx = build_grounding_context(ev)
    result = check_numeric_claims("batch size reduced from 32 to 8", ctx)
    assert result.ok, result.unverified_numbers


def test_colliding_stats_keys_do_not_silently_drop_a_token():
    # Regression: "flood-area-km2" and "flood_area_km2" both sanitize to the
    # token name "flood_area_km2" -- the second used to silently overwrite
    # the first in ctx.tokens (the value stayed in allowed_values as a set
    # member, but became unreachable as a named {placeholder}).
    from backend.shared.schemas import Confidence, Evidence, Modality, TaskType

    ev = Evidence(
        task=TaskType.OPTICAL_SAR_FUSION,
        model_used="m",
        modality_used=[Modality.OPTICAL],
        stats={"flood-area-km2": 3.2, "flood_area_km2": 9.9},
        confidence=Confidence(value=0.8, band="HIGH", basis="x"),
    )
    ctx = build_grounding_context(ev)
    values = set(ctx.tokens.values())
    assert "3.2" in values and "9.9" in values, ctx.tokens
