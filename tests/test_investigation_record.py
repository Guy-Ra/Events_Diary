"""Deterministic assembly tests for independent Investigation records."""

from dataclasses import replace
from pathlib import Path

import pytest

from events_diary.adaptive_sampling import AdaptiveSamplingPlan
from events_diary.config import ROI
from events_diary.evidence_candidate_timeline import EvidenceCandidate
from events_diary.frame_materialization import MaterializedEvidenceCandidate
from events_diary.fusion import (
    FinalInvestigationResult,
    FusedFieldObservation,
    FusedInvestigationResult,
    resolve_investigation,
)
from events_diary.investigation_lifecycle import (
    InvestigationEndReason,
    InvestigationInterval,
    InvestigationStartReason,
)
from events_diary.investigation_record import (
    VideoInvestigationResult,
    assemble_investigation_record,
)
from events_diary.investigation_window import InvestigationWindow
from events_diary.ocr_fields import extract_ocr_fields
from events_diary.representative_selection import (
    InvestigationSelection,
    RepresentativeSelection,
)
from events_diary.ticket_ocr import OCRDetection, OCRTicketResult
from events_diary.video import VideoMetadata
from events_diary.vlm import (
    RawVLMResult,
    VLMFieldObservation,
    VLMInvestigationResult,
    VLMParseDiagnostics,
)
from events_diary.vlm_review import VLMReviewObservation, VLMReviewResult

VIDEO = Path("data/videos/investigation_switch_test.mp4")


def _ocr(ticket_value, *, investigation_id, candidate_index, timestamp, field="REGISTRATION"):
    """Build one existing OCR occurrence with its original frame/time provenance."""
    roi = ROI(16, 121, 336, 831)
    detections = (
        OCRDetection(field, 0.99, ((26, 141), (126, 141), (126, 153), (26, 153))),
        OCRDetection(str(ticket_value), 0.98, ((26, 161), (86, 161), (86, 177), (26, 177))),
    )
    ticket = OCRTicketResult(
        "RapidOCR/test", "full_ticket_color_cubic_2x", roi, 2, detections,
        investigation_id, candidate_index, int(timestamp * 10), timestamp, "resolved",
    )
    return ticket, extract_ocr_fields(ticket)[0]


def _record(investigation_id, start, end, *, repeated_occurrence=False):
    """Build a complete typed record without invoking OCR or VLM runtimes."""
    interval = InvestigationInterval(
        investigation_id, start, end,
        InvestigationStartReason.VIDEO_STARTED_ACTIVE if investigation_id == 0 else InvestigationStartReason.SWITCH,
        InvestigationEndReason.SWITCH if investigation_id == 0 else InvestigationEndReason.VIDEO_END,
    )
    window = InvestigationWindow(investigation_id, start, end, start, end, 0.0, 0.0)
    timestamps = [start + (end - start) / 2]
    if repeated_occurrence:
        timestamps = [start + 0.5, end - 0.5]
    candidates = tuple(
        EvidenceCandidate(investigation_id, time, ("stable_ticket",), (), 0, False)
        for time in timestamps
    )
    materialized = tuple(
        MaterializedEvidenceCandidate(candidate, "resolved", int(time * 10), time, None, None, None)
        for candidate, time in zip(candidates, timestamps, strict=True)
    )
    plan = AdaptiveSamplingPlan(window, 10.0, 1.0, 10.0, 0.0, 0.0, (), tuple(timestamps))
    selection = InvestigationSelection(
        investigation_id,
        tuple(RepresentativeSelection(i, ("stable_ticket",)) for i in range(len(candidates))),
    )
    ocr_records, occurrences = [], []
    for index, timestamp in enumerate(timestamps):
        field = "TRACK" if repeated_occurrence else "REGISTRATION"
        raw = "281°" if index == 0 else "295°"
        result, occurrence = _ocr(
            raw if repeated_occurrence else "4X-ABC", investigation_id=investigation_id,
            candidate_index=index, timestamp=timestamp, field=field,
        )
        ocr_records.append(result)
        occurrences.append(occurrence)
    # A separate registration occurrence anchors the final field evidence.
    if repeated_occurrence:
        result, occurrence = _ocr(
            "4X-ABC", investigation_id=investigation_id, candidate_index=0,
            timestamp=timestamps[0],
        )
        ocr_records.append(result)
        occurrences.append(occurrence)

    raw_vlm = RawVLMResult(
        "success", investigation_id, VIDEO, VIDEO, start, end,
        "test-model", "revision-1", "fr24_vlm_v1", raw_text="{}",
    )
    vlm = VLMFieldObservation(
        "REGISTRATION", "4X-ABC", "observed", investigation_id,
        VIDEO, start, end, "test-model", "revision-1", "fr24_vlm_v1",
    )
    pass1 = VLMInvestigationResult(raw_vlm, (vlm,), VLMParseDiagnostics("valid_json"))
    registration = occurrences[-1]
    fused_field = FusedFieldObservation(
        "REGISTRATION", "agreement", registration, vlm,
        "4X-ABC", "4X-ABC", "4X-ABC",
    )
    fusion = FusedInvestigationResult(
        investigation_id, VIDEO, start, end, {
            "REGISTRATION": fused_field,
            "TRACK": FusedFieldObservation(
                "TRACK", "conflict", ocr_normalized_value="281", vlm_normalized_value="295",
            ),
            "AGE": FusedFieldObservation("AGE", "unavailable"),
        },
    )
    raw_review = RawVLMResult(
        "success", investigation_id, VIDEO, VIDEO, start, end,
        "test-model", "revision-1", "fr24_vlm_review_v1", raw_text="{}",
    )
    review = VLMReviewResult(raw_review, {
        "REGISTRATION": VLMReviewObservation("REGISTRATION", "4X-ABC", "confirmed", "clear"),
    })
    final = resolve_investigation(fusion, review)
    return {
        "investigation_id": investigation_id, "source_video": VIDEO,
        "interval": interval, "confirmed_events": (), "window": window,
        "sampling_plan": plan, "stable_views": (), "candidates": candidates,
        "materialized_candidates": materialized, "selection": selection,
        "ocr_results": tuple(ocr_records), "ocr_field_occurrences": tuple(occurrences),
        "vlm_pass1": pass1, "fusion_result": fusion, "vlm_pass2": review,
        "final_result": final,
    }


def _assemble(**overrides):
    """Create one assembled result from the common two-Investigation fixture."""
    values = _record(0, 0.0, 3.0)
    values.update(overrides)
    return assemble_investigation_record(**values)


def test_assembly_exposes_bounds_final_fields_and_source_provenance():
    record = _assemble()
    assert (record.start_time_s, record.end_time_s, record.duration_s) == (0.0, 3.0, 3.0)
    assert record.final_fields["REGISTRATION"].final_value == "4X-ABC"
    assert record.final_fields["REGISTRATION"].fused_observation.ocr_observation is not None
    assert record.final_fields["REGISTRATION"].vlm_review_observation.reviewed_value == "4X-ABC"
    assert record.unresolved_fields == ("TRACK",)
    assert record.not_available_fields == ("AGE",)
    with pytest.raises(TypeError):
        record.final_fields["AGE"] = record.final_fields["REGISTRATION"]


def test_multiple_dynamic_ocr_occurrences_are_preserved():
    values = _record(0, 0.0, 3.0, repeated_occurrence=True)
    record = assemble_investigation_record(**values)
    track = [item for item in record.ocr_field_occurrences if item.canonical_label == "TRACK"]
    assert [item.raw_value_text for item in track] == ["281°", "295°"]
    assert len(record.ocr_results) == 3


def test_two_same_aircraft_investigations_remain_independent_and_ordered():
    first = assemble_investigation_record(**_record(0, 0.0, 3.0))
    second = assemble_investigation_record(**_record(1, 3.0, 6.0))
    metadata = VideoMetadata(VIDEO, 6.0, 10.0, 60, 1920, 1080)
    session = VideoInvestigationResult(VIDEO, metadata, (first, second))
    assert session.investigation_count == 2
    assert [item.investigation_id for item in session.investigations] == [0, 1]
    assert first.final_fields is not second.final_fields
    assert first.ocr_field_occurrences[0].provenance.investigation_index == 0
    assert second.ocr_field_occurrences[0].provenance.investigation_index == 1
    assert first.final_fields["REGISTRATION"].final_value == second.final_fields["REGISTRATION"].final_value


def test_invalid_investigation_bounds_and_cross_investigation_evidence_are_rejected():
    values = _record(0, 0.0, 3.0)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        assemble_investigation_record(**{**values, "interval": replace(values["interval"], end_time_seconds=-1.0)})
    foreign = _record(1, 3.0, 6.0)["ocr_field_occurrences"][0]
    with pytest.raises(ValueError, match="different Investigation"):
        assemble_investigation_record(**{**values, "ocr_field_occurrences": (foreign,)})


def test_session_rejects_mismatched_video_metadata():
    record = assemble_investigation_record(**_record(0, 0.0, 3.0))
    other = VideoMetadata(Path("other.mp4"), 6.0, 10.0, 60, 1920, 1080)
    with pytest.raises(ValueError, match="metadata must describe"):
        VideoInvestigationResult(VIDEO, other, (record,))


def test_record_exposes_unresolved_and_not_available_field_views():
    record = _assemble()
    assert isinstance(record.final_result, FinalInvestigationResult)
    assert record.unresolved_fields == ("TRACK",)
    assert record.not_available_fields == ("AGE",)
