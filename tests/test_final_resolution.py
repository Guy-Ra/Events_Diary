"""Deterministic tests for conservative final OCR/VLM evidence resolution."""

from pathlib import Path

import pytest

from events_diary.config import ROI
from events_diary.fusion import (
    FinalFieldResult,
    FinalInvestigationResult,
    FinalInvestigationSummary,
    FusedFieldObservation,
    FusedInvestigationResult,
    normalize_field_value,
    resolve_field,
    resolve_investigation,
)
from events_diary.ocr_fields import extract_ocr_fields
from events_diary.ticket_ocr import OCRDetection, OCRTicketResult
from events_diary.vlm import VLMFieldObservation
from events_diary.vlm_review import VLMReviewObservation

VIDEO = Path("data/videos/session.mp4")


def review(value, *, status="confirmed", quality="clear", time=None, name="REGISTRATION"):
    """Build a review record for the REGISTRATION canonical field."""
    return VLMReviewObservation(name, value, status, quality, time)


def fused(status, ocr="AB-123", vlm="AB-123", *, name="REGISTRATION", time=None):
    """Build fused evidence with normalized values and optional source records."""
    ocr_record = None
    vlm_record = None
    if time is not None:
        detections = (
            OCRDetection("SQUAWK", .99, ((26, 141), (126, 141), (126, 153), (26, 153))),
            OCRDetection(str(ocr), .98, ((26, 161), (86, 161), (86, 177), (26, 177))),
        )
        ticket = OCRTicketResult("test", "test", ROI(16, 121, 336, 831), 2, detections, 2, 7, 40, time, "resolved")
        ocr_record = next(item for item in extract_ocr_fields(ticket) if item.canonical_label == "SQUAWK")
        vlm_record = VLMFieldObservation(
            "SQUAWK", vlm, "observed", 2, VIDEO, 0, 8.4, "test", None, "test", evidence_time_s=time,
        )
        name = "SQUAWK"
    return FusedFieldObservation(
        name, status, ocr_observation=ocr_record, vlm_observation=vlm_record,
        ocr_normalized_value=ocr, vlm_normalized_value=vlm,
    )


@pytest.mark.parametrize(
    ("fusion_status", "ocr", "vlm", "pass2", "review_status", "quality", "expected_status", "expected"),
    [
        ("agreement", "AB-123", "AB-123", "ab-123", "confirmed", "clear", "resolved", "AB-123"),
        ("agreement", "16 years", "16 years", "24 years", "corrected", "clear", "unresolved", None),
        ("agreement", "AB-123", "AB-123", None, "unresolved", "weak", "resolved", "AB-123"),
        ("agreement", "AB-123", "AB-123", None, "not_visible", "not_visible", "resolved", "AB-123"),
        ("conflict", "1100", "1130", "1100", "confirmed", "clear", "resolved", "1100"),
        ("conflict", "1100", "1130", "1130", "confirmed", "clear", "resolved", "1130"),
        ("conflict", "1100", "1130", "1180", "corrected", "clear", "unresolved", None),
        ("conflict", "1100", "1130", "1130", "confirmed", "partial", "unresolved", None),
        ("conflict", "1100", "1130", "1130", "confirmed", "weak", "unresolved", None),
        ("conflict", "1100", "1130", None, "not_visible", "not_visible", "unresolved", None),
        ("ocr_only", "AB-123", None, "ab-123", "confirmed", "clear", "resolved_single_source", "AB-123"),
        ("ocr_only", "AB-123", None, "CD-456", "corrected", "clear", "unresolved", None),
        ("ocr_only", "AB-123", None, None, "unresolved", "weak", "unresolved", None),
        ("vlm_only", None, "AB-123", "ab-123", "confirmed", "clear", "resolved_single_source", "AB-123"),
        ("vlm_only", None, "AB-123", "CD-456", "corrected", "clear", "unresolved", None),
        ("unavailable", None, None, None, "unresolved", "weak", "not_available", None),
        ("unavailable", None, None, "ab-123", "corrected", "clear", "resolved_single_source", "AB-123"),
    ],
)
def test_baseline_resolution_rules(
    fusion_status, ocr, vlm, pass2, review_status, quality, expected_status, expected,
):
    """Exercise agreement, conflict, single-source, and unavailable policies."""
    result = resolve_field(
        fused(fusion_status, ocr, vlm),
        review(pass2, status=review_status, quality=quality),
    )
    assert (result.final_status, result.final_value) == (expected_status, expected)
    assert result.resolution_reason


@pytest.mark.parametrize(
    ("pass2", "expected"), [("1200", "1200"), ("1300", "1300")],
)
def test_temporal_mismatch_resolves_only_at_comparable_point(pass2, expected):
    """Resolve a dynamic mismatch against a point-timed candidate only."""
    result = resolve_field(fused("temporal_mismatch", "1200", "1300", time=4.0), review(pass2, time=4.0, name="SQUAWK"))
    assert result.final_status == "resolved"
    assert result.final_value == expected
    assert result.temporal_status == "comparable"


@pytest.mark.parametrize("time", [None, 4.1])
def test_temporal_mismatch_rejects_missing_or_noncomparable_review_time(time):
    """Do not let an untimed or differently timed review decide a dynamic field."""
    result = resolve_field(fused("temporal_mismatch", "1200", "1300", time=4.0), review("1200", time=time, name="SQUAWK"))
    assert result.final_status == "unresolved"
    assert result.final_value is None


def test_temporal_vlm1_candidate_requires_its_own_matching_point_time():
    """A VLM1 value resolves temporal conflict only when tied to the same point."""
    evidence = fused("temporal_mismatch", "1200", "1300", time=4.0)
    evidence = FusedFieldObservation(
        **{**evidence.__dict__, "vlm_observation": VLMFieldObservation(
            "SQUAWK", "1300", "observed", 2, VIDEO, 0, 8.4, "test", None, "test", evidence_time_s=4.0,
        )},
    )
    assert resolve_field(evidence, review("1300", time=4.0, name="SQUAWK")).final_value == "1300"


def test_boolean_false_and_leading_zero_normalization_are_preserved():
    """Retain false as usable data and normalize identifiers without dropping zeros."""
    boolean = FusedFieldObservation("scroll_observed", "agreement", ocr_normalized_value=False, vlm_normalized_value=False)
    reviewed = VLMReviewObservation("scroll_observed", False, "confirmed", "clear")
    result = resolve_field(boolean, reviewed)
    assert (result.final_status, result.final_value) == ("resolved", False)
    identifier = resolve_field(fused("conflict", "0012", "0013"), review("0012"))
    assert identifier.final_value == "0012"


def test_pass2_normalization_preserves_raw_value_and_sources_are_immutable():
    """Normalize only for comparison; retain exact source text and records."""
    evidence = fused("agreement", normalize_field_value("REGISTRATION", "ab-123"), "AB-123")
    raw = review("  ab-123  ")
    old = (evidence.ocr_normalized_value, evidence.vlm_normalized_value, raw.reviewed_value)
    result = resolve_field(evidence, raw)
    assert result.final_value == "AB-123"
    assert raw.reviewed_value == "  ab-123  "
    assert old == (evidence.ocr_normalized_value, evidence.vlm_normalized_value, raw.reviewed_value)


def test_no_majority_vote_when_review_challenges_prior_agreement():
    """Two matching sources cannot outvote a distinct Pass 2 challenge."""
    result = resolve_field(fused("agreement", "16 years", "16 years"), review("24 years", status="corrected"))
    assert (result.final_status, result.final_value) == ("unresolved", None)


def test_agreement_with_weak_matching_review_keeps_no_reconfirmation_claim():
    """Weak review quality cannot be presented as an independent confirmation."""
    result = resolve_field(
        fused("agreement"), review("AB-123", quality="weak"),
    )
    assert result.final_status == "unresolved"
    assert "could not clearly confirm" in result.resolution_reason


def test_conflict_with_duplicate_candidates_is_not_exactly_one_match():
    """A purported conflict with indistinguishable candidates is never auto-picked."""
    result = resolve_field(
        fused("conflict", "AB-123", "AB-123"), review("AB-123"),
    )
    assert (result.final_status, result.final_value) == ("unresolved", None)


def test_ocr_only_not_visible_review_remains_unresolved():
    """An unavailable second pass cannot confirm the lone OCR candidate."""
    result = resolve_field(
        fused("ocr_only", "AB-123", None),
        review(None, status="not_visible", quality="not_visible"),
    )
    assert (result.final_status, result.final_value) == ("unresolved", None)


def test_investigation_resolution_summary_and_read_only_field_map():
    """Aggregate statuses accurately and expose an immutable final field map."""
    fields = {
        "REGISTRATION": fused("agreement"),
        "SERIAL_NUMBER": fused("unavailable", name="SERIAL_NUMBER"),
    }
    source = FusedInvestigationResult(2, VIDEO, 0, 8.4, fields)
    final = resolve_investigation(source)
    assert final.summary == FinalInvestigationSummary(2, 1, 0, 0, 1)
    assert isinstance(final, FinalInvestigationResult)
    with pytest.raises(TypeError):
        final.fields["AGE"] = final.fields["REGISTRATION"]


def test_final_field_rejects_invalid_status():
    """Runtime validation rejects statuses outside the documented literals."""
    evidence = fused("agreement")
    with pytest.raises(ValueError, match="final_status"):
        FinalFieldResult("REGISTRATION", "winner", "AB-123", evidence, None, "invalid")
