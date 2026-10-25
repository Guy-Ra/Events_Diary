"""Deterministic assembly tests for independent Investigation records."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from events_diary.adaptive_sampling import AdaptiveSamplingPlan
from events_diary.canonical_events import (
    CanonicalAircraft,
    CanonicalDataQuality,
    CanonicalEvent,
    CanonicalFlight,
    CanonicalMovement,
    CanonicalRoute,
    CanonicalSession,
    CanonicalSourceVideo,
    build_canonical_event,
    build_canonical_session,
)
from events_diary.config import ROI
from events_diary.evidence_candidate_timeline import EvidenceCandidate
from events_diary.frame_materialization import MaterializedEvidenceCandidate
from events_diary.fusion import (
    FinalFieldResult,
    FinalInvestigationResult,
    FinalInvestigationSummary,
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
    InvestigationFailure,
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
    VLMVisualRouteClaim,
    VLMVisualRouteObservation,
)
from events_diary.vlm_review import (
    VLMReviewObservation,
    VLMReviewResult,
    VLMRouteReviewObservation,
)

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


def _video_metadata(path=VIDEO, duration=8.1):
    """Build authoritative source metadata for canonical session tests."""
    return VideoMetadata(path, duration, 10.0, round(duration * 10), 1920, 1080)


def _video_result(records, metadata=None):
    """Wrap complete Phase 6 records in its existing per-video result contract."""
    metadata = metadata or _video_metadata()
    return VideoInvestigationResult(metadata.path, metadata, tuple(records))


def _record_with_fields(record, specs):
    """Add representative final-field results while preserving record contracts."""
    final_fields = dict(record.final_result.fields)
    fused_fields = dict(record.fusion_result.fields)
    for name, status, value in specs:
        fused = FusedFieldObservation(name, "unavailable")
        fused_fields[name] = fused
        final_fields[name] = FinalFieldResult(name, status, value, fused, None, "test fixture")
    counts = {
        status: sum(item.final_status == status for item in final_fields.values())
        for status in ("resolved", "resolved_single_source", "unresolved", "not_available")
    }
    summary = FinalInvestigationSummary(
        len(final_fields), counts["resolved"], counts["resolved_single_source"],
        counts["unresolved"], counts["not_available"],
    )
    fusion = replace(record.fusion_result, fields=fused_fields)
    final = replace(record.final_result, fields=final_fields, summary=summary)
    return replace(record, fusion_result=fusion, final_result=final)


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


def test_record_exposes_derived_track_directions_linked_to_original_ocr_occurrences():
    values = _record(0, 0.0, 3.0, repeated_occurrence=True)
    record = assemble_investigation_record(**values)
    directions = record.derived_track_directions
    assert [item.source_observation.raw_value_text for item in directions] == ["281°", "295°"]
    assert [item.provenance for item in directions] == [
        "deterministic_derived_from_ocr_track", "deterministic_derived_from_ocr_track",
    ]
    assert [item.status for item in record.track_direction_cross_checks] == [
        "unavailable", "unavailable",
    ]


def test_canonical_event_maps_final_values_and_keeps_nested_contracts_immutable():
    """Create one event from Phase 6 final fields with the approved nested groups."""
    event = build_canonical_event(_assemble(), event_id=0)
    assert isinstance(event, CanonicalEvent)
    assert event.investigation_id == 0
    assert (event.event_time_s, event.start_time_s, event.end_time_s) == (0.0, 0.0, 3.0)
    assert event.aircraft.registration == "4X-ABC"
    assert event.route.origin_airport_code is None
    assert "route.origin_airport_code" not in event.data_quality.not_available_fields
    assert "movement.track_degrees" in event.data_quality.unresolved_fields
    assert set(event.to_dict()) == {
        "event_id", "investigation_id", "event_time_s", "start_time_s", "end_time_s",
        "flight", "route", "aircraft", "movement", "data_quality",
    }
    with pytest.raises(AttributeError):
        event.event_id = 2


def test_canonical_mapping_propagates_resolved_and_resolved_single_source_values():
    """Map final Phase 6 values without exposing upstream field names."""
    record = _record_with_fields(_assemble(), [
        ("primary_flight_id", "resolved_single_source", "MEA251"),
        ("alternate_flight_id", "resolved", "ME251"),
        ("airline_name", "resolved", "Middle East Airlines"),
        ("origin_airport_code", "resolved", "BEY"),
        ("destination_airport_code", "resolved", "ATH"),
        ("origin_scheduled_time", "resolved", "09:10"),
        ("origin_actual_time", "resolved", "09:18"),
        ("destination_scheduled_time", "resolved", "11:05"),
        ("destination_estimated_or_actual_time", "resolved", "11:05"),
        ("aircraft_code", "resolved", "A21N"),
        ("AIRCRAFT_TYPE", "resolved", "Airbus A321-271NX"),
        ("TRACK", "resolved", "293"),
    ])
    event = build_canonical_event(record, event_id=4)
    assert event.flight.primary_id == "MEA251"
    assert event.flight.alternate_id == "ME251"
    assert event.flight.airline == "Middle East Airlines"
    assert event.route.origin_airport_code == "BEY"
    assert event.route.destination_airport_code == "ATH"
    assert event.route.scheduled_departure == "09:10"
    assert event.route.actual_departure == "09:18"
    assert event.route.scheduled_arrival == "11:05"
    assert event.route.estimated_or_actual_arrival == "11:05"
    assert event.aircraft.code == "A21N"
    assert event.aircraft.type == "Airbus A321-271NX"
    assert event.movement.track_degrees == 293.0
    assert "flight.primary_id" not in event.data_quality.unresolved_fields
    assert "route.origin_airport_code" not in event.data_quality.not_available_fields


def test_canonical_movement_carries_phase6_direction_and_visual_claims():
    """Prefer final Pass 2 visual reviews over retained Pass 1 claims."""
    record = assemble_investigation_record(
        **_record(0, 0.0, 3.0, repeated_occurrence=True),
    )
    record = _record_with_fields(record, [("TRACK", "resolved", "281")])
    consistency = record.track_direction_cross_checks[0]
    record = replace(
        record,
        final_result=replace(record.final_result, direction_consistency=consistency),
    )
    route_claims = VLMVisualRouteObservation(
        direction_sector=VLMVisualRouteClaim("NW", "observed"),
        direction_text=VLMVisualRouteClaim("Northwest", "observed"),
        route_description=VLMVisualRouteClaim("Route extends northwest", "inferred"),
        departure_context=VLMVisualRouteClaim("Near BEY", "observed"),
        investigation_index=0,
        source_video=VIDEO,
        source_start_s=0.0,
        source_end_s=3.0,
        model_name=record.vlm_pass1.model_name,
        model_revision=record.vlm_pass1.model_revision,
        prompt_version=record.vlm_pass1.prompt_version,
    )
    record = replace(record, vlm_pass1=replace(record.vlm_pass1, visual_route=route_claims))
    final_reviews = {
        "visual_flight_direction_sector": VLMRouteReviewObservation(
            "visual_flight_direction_sector", "NW", "confirmed", "clear", "observed",
        ),
        "visual_flight_direction_text": VLMRouteReviewObservation(
            "visual_flight_direction_text", "Northwest reviewed", "corrected", "clear", "observed",
        ),
        "visual_route_description": VLMRouteReviewObservation(
            "visual_route_description", "Reviewed route description", "corrected", "clear", "inferred",
        ),
        "visual_departure_context": VLMRouteReviewObservation(
            "visual_departure_context", "Reviewed near BEY", "confirmed", "clear", "observed",
        ),
    }
    record = replace(record, vlm_pass2=replace(record.vlm_pass2, route_reviews=final_reviews))
    event = build_canonical_event(record, event_id=0)
    assert event.movement.track_degrees == 281.0
    assert event.movement.direction == "West"
    assert event.movement.visual_flight_direction == "Northwest reviewed"
    assert event.movement.visual_route_description == "Reviewed route description"
    assert event.movement.visual_departure_context == "Reviewed near BEY"


def test_canonical_visual_review_statuses_distinguish_unresolved_unavailable_and_absent():
    """Map only explicit final visual statuses and leave missing reviews absent."""
    record = _assemble()
    reviews = {
        "visual_flight_direction_text": VLMRouteReviewObservation(
            "visual_flight_direction_text", None, "unresolved", "weak", "not_visible",
        ),
        "visual_route_description": VLMRouteReviewObservation(
            "visual_route_description", None, "not_visible", "not_visible", "not_visible",
        ),
    }
    record = replace(record, vlm_pass2=replace(record.vlm_pass2, route_reviews=reviews))
    event = build_canonical_event(record, event_id=0)
    assert event.field_state("movement.visual_flight_direction") == "unresolved"
    assert event.field_state("movement.visual_route_description") == "not_available"
    assert event.field_state("movement.visual_departure_context") == "absent"


def test_canonical_unresolved_and_not_available_values_are_null_with_paths():
    """Retain Phase 6 unresolved and unavailable classifications as canonical paths."""
    record = _record_with_fields(_assemble(), [
        ("alternate_flight_id", "unresolved", None),
        ("destination_airport_code", "not_available", None),
    ])
    event = build_canonical_event(record, event_id=5)
    assert event.flight.alternate_id is None
    assert event.route.destination_airport_code is None
    assert "flight.alternate_id" in event.data_quality.unresolved_fields
    assert "route.destination_airport_code" in event.data_quality.not_available_fields
    assert event.field_state("flight.alternate_id") == "unresolved"
    assert event.field_state("route.destination_airport_code") == "not_available"


def test_canonical_event_time_and_interval_are_preserved():
    """Use Investigation start as semantic event time and preserve both bounds."""
    event = build_canonical_event(_assemble(), event_id=1)
    assert event.event_time_s == event.start_time_s == 0.0
    assert event.end_time_s == 3.0
    with pytest.raises(ValueError, match="event_time_s"):
        CanonicalEvent(1, 0, 4.0, 0.0, 3.0)
    with pytest.raises(ValueError, match="must not precede"):
        CanonicalEvent(1, 0, 1.0, 2.0, 1.0)


def test_canonical_session_sorts_by_event_time_then_investigation_id_and_derives_count():
    """Order records first, then assign sequential IDs with Investigation-ID ties."""
    late = _assemble(**_record(0, 3.0, 6.0))
    tied_high = _assemble(**_record(2, 1.0, 2.0))
    tied_low = _assemble(**_record(1, 1.0, 2.0))
    session = build_canonical_session(
        _video_result([tied_high, tied_low, late], _video_metadata(duration=8.0)),
    )
    assert session.schema_version == "1.0"
    assert [event.investigation_id for event in session.events] == [1, 2, 0]
    assert [event.event_id for event in session.events] == [0, 1, 2]
    assert session.events_count == len(session.events) == 3
    assert session.source_video == CanonicalSourceVideo(VIDEO.name, 8.0)


def test_canonical_session_accepts_empty_and_single_record_inputs():
    """Represent a successfully processed empty video and ordinary single event."""
    metadata = _video_metadata()
    empty = build_canonical_session(_video_result([], metadata))
    assert empty.events == ()
    assert empty.events_count == 0
    one = build_canonical_session(_video_result([_assemble()], metadata))
    assert len(one.events) == one.events_count == 1
    assert (one.events[0].event_id, one.events[0].investigation_id) == (0, 0)


def test_phase6_video_result_rejects_records_from_a_different_source_video():
    """Rely on Phase 6's existing source-consistency contract at the boundary."""
    with pytest.raises(ValueError, match="share source_video"):
        VideoInvestigationResult(Path("other.mp4"), _video_metadata(Path("other.mp4")), (_assemble(),))


def test_canonical_aggregation_rejects_video_results_with_failed_investigations():
    """Do not produce a seemingly complete session while detected work failed."""
    failure = InvestigationFailure(
        1, VIDEO, 4.0, 7.0, "ocr", "fixture failure", Path("debug/investigation_001"),
    )
    result = VideoInvestigationResult(VIDEO, _video_metadata(), (), (failure,))
    with pytest.raises(ValueError, match="failed Investigations"):
        build_canonical_session(result)


def test_canonical_aggregation_keeps_quality_local_and_does_not_mutate_inputs():
    """Create independent event groups and preserve each input record unchanged."""
    first = _record_with_fields(_assemble(), [
        ("alternate_flight_id", "unresolved", None),
    ])
    second = _record_with_fields(_assemble(**_record(1, 4.0, 7.0)), [
        ("destination_airport_code", "not_available", None),
        ("REGISTRATION", "resolved", "9H-XYZ"),
    ])
    before = (dict(first.final_result.fields), dict(second.final_result.fields))
    session = build_canonical_session(_video_result([first, second]))
    assert (dict(first.final_result.fields), dict(second.final_result.fields)) == before
    assert "flight.alternate_id" in session.events[0].data_quality.unresolved_fields
    assert "flight.alternate_id" not in session.events[1].data_quality.unresolved_fields
    assert "route.destination_airport_code" in session.events[1].data_quality.not_available_fields
    assert "route.destination_airport_code" not in session.events[0].data_quality.not_available_fields
    assert session.events[0].aircraft.registration == "4X-ABC"
    assert session.events[1].aircraft.registration == "9H-XYZ"
    assert session.events[0].data_quality is not session.events[1].data_quality
    assert session.events[0].flight is not session.events[1].flight
    assert session.events[0].event_id == 0 and session.events[1].event_id == 1


def test_canonical_session_rejects_duplicate_investigations_and_fixed_version():
    """Keep one event per Investigation and the external schema version fixed."""
    first = build_canonical_event(_assemble(), event_id=0)
    duplicate = build_canonical_event(_assemble(), event_id=1)
    with pytest.raises(ValueError, match="exactly one"):
        CanonicalSession(CanonicalSourceVideo("session.mp4", 3.0), (first, duplicate))
    with pytest.raises(ValueError, match="schema_version"):
        CanonicalSession(CanonicalSourceVideo("session.mp4", 3.0), (), schema_version="2.0")


def test_canonical_json_has_approved_shape_and_no_phase6_internals_or_excluded_fields():
    """Serialize clean nested data while excluding evidence, provider, and debug objects."""
    session = build_canonical_session(_video_result([
        _assemble(), _assemble(**_record(1, 4.0, 7.0)),
    ]))
    output = json.loads(session.to_json())
    assert list(output) == ["schema_version", "source_video", "events_count", "events"]
    assert output["schema_version"] == "1.0"
    assert output["source_video"] == {"name": "investigation_switch_test.mp4", "duration_s": 8.1}
    assert output["events_count"] == 2
    canonical = output["events"][0]
    assert canonical["aircraft"]["registration"] == "4X-ABC"
    assert [item["event_id"] for item in output["events"]] == [0, 1]
    assert "REGISTRATION" not in json.dumps(output)
    for excluded in (
        "SERIAL_NUMBER", "AGE", "MACH", "TEMPERATURE", "FIR_UIR", "LATITUDE",
        "LONGITUDE", "ocr_field_occurrences", "vlm_pass1", "vlm_pass2", "fusion_result",
        "provider_metadata", "diagnostics", "resolution_reason",
    ):
        assert excluded not in json.dumps(output)


def test_canonical_quality_paths_reject_phase6_names_and_overlapping_statuses():
    """Limit quality lists to canonical paths and disallow contradictory labels."""
    with pytest.raises(ValueError, match="unknown canonical"):
        CanonicalDataQuality(unresolved_fields=("REGISTRATION",))
    with pytest.raises(ValueError, match="both unresolved and unavailable"):
        CanonicalDataQuality(
            unresolved_fields=("aircraft.registration",),
            not_available_fields=("aircraft.registration",),
        )
    with pytest.raises(ValueError, match="duplicate"):
        CanonicalDataQuality(unresolved_fields=("flight.airline", "flight.airline"))
    quality = CanonicalDataQuality(
        unresolved_fields=("flight.primary_id", "aircraft.registration"),
        not_available_fields=("route.actual_departure", "flight.alternate_id"),
    )
    assert quality.unresolved_fields == ("aircraft.registration", "flight.primary_id")
    assert quality.not_available_fields == ("flight.alternate_id", "route.actual_departure")


def test_canonical_field_state_covers_value_and_absence_and_rejects_unknown_path():
    """Expose renderer-ready states without adding them to serialized data."""
    event = build_canonical_event(_assemble(), event_id=0)
    assert event.field_state("aircraft.registration") == "value"
    assert event.field_state("flight.alternate_id") == "absent"
    assert "field_state" not in event.to_dict()
    with pytest.raises(ValueError, match="known canonical field"):
        event.field_state("REGISTRATION")


def test_canonical_event_rejects_non_null_values_with_either_quality_status():
    """Keep stored values and their separate quality classifications consistent."""
    event = build_canonical_event(_assemble(), event_id=0)
    for quality in (
        CanonicalDataQuality(unresolved_fields=("aircraft.registration",)),
        CanonicalDataQuality(not_available_fields=("aircraft.registration",)),
    ):
        with pytest.raises(ValueError, match="Non-null canonical field"):
            replace(event, data_quality=quality)


def test_canonical_json_keeps_null_values_and_quality_semantics_machine_clean():
    """Serialize unresolved and unavailable fields as null plus canonical paths."""
    record = _record_with_fields(_assemble(), [
        ("alternate_flight_id", "unresolved", None),
        ("destination_airport_code", "not_available", None),
    ])
    payload = build_canonical_event(record, event_id=0).to_dict()
    assert payload["flight"]["alternate_id"] is None
    assert payload["route"]["destination_airport_code"] is None
    assert payload["data_quality"] == {
        "unresolved_fields": [
            "flight.alternate_id", "movement.direction", "movement.track_degrees",
        ],
        "not_available_fields": ["route.destination_airport_code"],
    }
    assert "N/A" not in json.dumps(payload)
    assert "Unresolved" not in json.dumps(payload)


def test_canonical_nested_contracts_validate_optional_values_and_source_metadata():
    """Reject invalid scalar values and preserve immutable source metadata."""
    assert CanonicalFlight(primary_id="MEA251").primary_id == "MEA251"
    assert CanonicalRoute(origin_airport_code="BEY").origin_airport_code == "BEY"
    assert CanonicalAircraft(registration="T7-ME2").registration == "T7-ME2"
    assert CanonicalMovement(track_degrees=293.0, direction="West-Northwest").direction
    with pytest.raises(ValueError, match=r"within \[0, 360\]"):
        CanonicalMovement(track_degrees=361)
    with pytest.raises(ValueError, match="duration_s"):
        CanonicalSourceVideo("session.mp4", float("inf"))
