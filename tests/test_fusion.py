"""Focused tests for immutable OCR/VLM fusion data contracts."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MappingProxyType

import pytest

from events_diary.config import ROI
from events_diary.fusion import (
    BOOLEAN_FIELDS,
    DYNAMIC_FIELDS,
    FIELD_FAMILIES,
    HEADING_FIELDS,
    IDENTIFIER_FIELDS,
    NUMERIC_FIELDS,
    STATIC_FIELDS,
    TEXT_FIELDS,
    TIME_FIELDS,
    FusedFieldObservation,
    FusedInvestigationResult,
    FusionSummary,
    apply_temporal_awareness,
    compare_field_observations,
    is_temporally_comparable,
    normalize_field_value,
)
from events_diary.ocr_fields import extract_ocr_fields
from events_diary.ticket_ocr import OCRDetection, OCRTicketResult
from events_diary.vlm import VLMFieldObservation

SOURCE_VIDEO = Path("data/videos/session.mp4")


def ocr_observation(value="4X-ABC", *, include_value=True, timestamp=4.0):
    """Build a typed OCR field record without invoking an OCR runtime."""
    roi = ROI(16, 121, 336, 831)
    detections = [
        OCRDetection("REGISTRATION", .99, ((26, 141), (126, 141), (126, 153), (26, 153))),
    ]
    if include_value:
        detections.append(
            OCRDetection(value, .98, ((26, 161), (86, 161), (86, 177), (26, 177)))
        )
    ticket = OCRTicketResult(
        "RapidOCR/test", "full_ticket_color_cubic_2x", roi, 2, tuple(detections),
        2, 7, 40, timestamp, "resolved",
    )
    fields = extract_ocr_fields(ticket)
    return next(field for field in fields if field.canonical_label == "REGISTRATION")


def vlm_observation(
    value="4X-ABC", *, status="observed", field_name="REGISTRATION",
    evidence_time_s=None,
):
    """Build a valid VLM field record with source provenance."""
    return VLMFieldObservation(
        field_name=field_name,
        raw_value=value,
        status=status,
        investigation_index=2,
        source_video=SOURCE_VIDEO,
        source_start_s=0.0,
        source_end_s=8.4,
        model_name="gemini-flash-latest",
        model_revision="pinned-revision",
        prompt_version="fr24_vlm_v1",
        evidence_time_s=evidence_time_s,
    )


def fused_field(**overrides):
    """Create a basic fused field, allowing test-specific contract values."""
    values = {"field_name": "REGISTRATION", "status": "agreement"}
    values.update(overrides)
    return FusedFieldObservation(**values)


def test_fused_field_preserves_source_records_and_separates_value_layers():
    ocr = ocr_observation()
    vlm = vlm_observation()
    original_ocr_value = ocr.raw_value_text
    original_vlm_value = vlm.raw_value
    result = fused_field(
        ocr_observation=ocr,
        vlm_observation=vlm,
        ocr_normalized_value="4XABC",
        vlm_normalized_value="4XABC",
        resolved_value=None,
        diagnostics=["Raw source text retained."],
    )
    assert result.ocr_observation is ocr
    assert result.vlm_observation is vlm
    assert result.ocr_normalized_value == result.vlm_normalized_value == "4XABC"
    assert result.resolved_value is None
    assert result.diagnostics == ("Raw source text retained.",)
    assert ocr.raw_value_text == original_ocr_value
    assert vlm.raw_value == original_vlm_value
    with pytest.raises(FrozenInstanceError):
        result.resolved_value = "changed"


@pytest.mark.parametrize(
    "status",
    ["agreement", "conflict", "ocr_only", "vlm_only", "unavailable", "temporal_mismatch"],
)
def test_all_supported_status_literals_are_accepted(status):
    assert fused_field(status=status).status == status


def test_unsupported_status_is_rejected():
    with pytest.raises(ValueError, match="Unsupported fusion status"):
        fused_field(status="pending")


def test_investigation_preserves_bounds_and_exposes_read_only_field_mapping():
    field = fused_field()
    summary = FusionSummary(total_fields=1, agreement_count=1)
    result = FusedInvestigationResult(
        investigation_id="investigation-2",
        source_video=SOURCE_VIDEO,
        start_time_s=12.0,
        end_time_s=20.4,
        fields={"REGISTRATION": field},
        summary=summary,
        diagnostics=("Contract-only result.",),
    )
    assert result.fields["REGISTRATION"] is field
    assert isinstance(result.fields, MappingProxyType)
    assert (result.source_video, result.start_time_s, result.end_time_s) == (
        SOURCE_VIDEO, 12.0, 20.4
    )
    assert result.summary is summary
    assert result.diagnostics == ("Contract-only result.",)
    with pytest.raises(TypeError):
        result.fields["TRACK"] = fused_field(field_name="TRACK")


def test_investigation_rejects_invalid_time_ranges():
    with pytest.raises(ValueError, match="end_time_s"):
        FusedInvestigationResult("id", SOURCE_VIDEO, 5.0, 4.0, {})
    with pytest.raises(ValueError, match="finite and nonnegative"):
        FusedInvestigationResult("id", SOURCE_VIDEO, float("nan"), 4.0, {})


def test_investigation_rejects_mismatched_field_mapping_key():
    with pytest.raises(ValueError, match="mapping key must match"):
        FusedInvestigationResult(
            "id", SOURCE_VIDEO, 0.0, 1.0, {"TRACK": fused_field()}
        )


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_summary_counts_must_be_nonnegative_integers(count):
    with pytest.raises((TypeError, ValueError)):
        FusionSummary(total_fields=count)


def test_summary_counts_are_preserved_and_total_must_match_fields():
    summary = FusionSummary(
        total_fields=2,
        agreement_count=1,
        conflict_count=1,
        ocr_only_count=0,
        vlm_only_count=0,
        unavailable_count=0,
        temporal_mismatch_count=0,
    )
    assert summary.total_fields == 2
    assert summary.conflict_count == 1
    with pytest.raises(ValueError, match="total_fields"):
        FusedInvestigationResult(
            "id", SOURCE_VIDEO, 0.0, 1.0, {"REGISTRATION": fused_field()}, summary
        )


def test_contract_does_not_require_a_summary():
    result = FusedInvestigationResult("id", SOURCE_VIDEO, 0.0, 1.0, {})
    assert result.summary is None


@pytest.mark.parametrize(
    "field_name,raw,expected",
    [
        ("REGISTRATION", " 4x-bal ", "4X-BAL"),
        ("origin_airport_code", " jfk ", "JFK"),
        ("primary_flight_id", " isr972 ", "ISR972"),
        ("SERIAL_NUMBER", " 001100 ", "001100"),
        ("ICAO_ADDRESS", " ab12cd ", "AB12CD"),
    ],
)
def test_identifier_cleanup_trims_collapses_and_uppercases(field_name, raw, expected):
    assert normalize_field_value(field_name, raw) == expected


@pytest.mark.parametrize(
    "field_name,raw,expected",
    [
        ("BAROMETRIC_ALTITUDE", " 5212 ", "5212"),
        ("GPS_ALTITUDE", " 1 130 ", "1 130"),
        ("MACH", "0.80", "0.80"),
        ("SQUAWK", "0012", "0012"),
    ],
)
def test_numeric_cleanup_preserves_digits_and_numeric_format(field_name, raw, expected):
    assert normalize_field_value(field_name, raw) == expected


@pytest.mark.parametrize("raw", ["281°", " 281 ° ", "281"])
def test_heading_normalization_removes_degree_formatting_only(raw):
    assert normalize_field_value("TRACK", raw) == "281"


@pytest.mark.parametrize(
    "raw,expected",
    [(True, True), (False, False), ("true", True), ("TRUE", True), ("false", False), (" FALSE ", False)],
)
def test_explicit_boolean_forms_normalize_deterministically(raw, expected):
    assert normalize_field_value("scroll_observed", raw) is expected


def test_unrecognized_boolean_text_is_not_interpreted_as_boolean():
    assert normalize_field_value("same_aircraft_throughout", "yes") == "yes"


def test_text_whitespace_is_collapsed_without_case_or_word_changes():
    assert normalize_field_value("airline_name", "  Israir   Airlines  ") == "Israir Airlines"
    assert normalize_field_value("AIRCRAFT_TYPE", "Airbus A330-243") == "Airbus A330-243"


def test_time_cleanup_preserves_semantics_and_formatting():
    assert normalize_field_value("origin_scheduled_time", " 2:30   PM ") == "2:30 PM"
    assert normalize_field_value("origin_scheduled_time", "2:30 PM") != "14:30"


def test_none_unknown_fields_and_source_independence():
    assert normalize_field_value("TRACK", None) is None
    value = "  MiXeD   Unknown Text  "
    assert normalize_field_value("unregistered_future_field", value) == "MiXeD Unknown Text"
    assert normalize_field_value("REGISTRATION", " 4x-bal ") == normalize_field_value(
        "REGISTRATION", " 4x-bal "
    )
    assert value == "  MiXeD   Unknown Text  "


def test_normalizing_source_values_does_not_mutate_source_observations():
    ocr = ocr_observation()
    vlm = vlm_observation()
    before_ocr = ocr.raw_value_text
    before_vlm = vlm.raw_value
    assert normalize_field_value("REGISTRATION", before_ocr) == before_ocr
    assert normalize_field_value("REGISTRATION", vlm.raw_value) == "4X-ABC"
    assert ocr.raw_value_text == before_ocr
    assert vlm.raw_value == before_vlm


@pytest.mark.parametrize(
    "field_name,raw,expected",
    [
        ("SERIAL_NUMBER", "1100", "1100"),
        ("AIRCRAFT_TYPE", "A320", "A320"),
        ("origin_airport_code", "John F Kennedy International Airport", "JOHN F KENNEDY INTERNATIONAL AIRPORT"),
        ("airline_name", "Israir Airlnes", "Israir Airlnes"),
    ],
)
def test_normalization_does_not_repair_or_infer_values(field_name, raw, expected):
    normalized = normalize_field_value(field_name, raw)
    assert normalized == expected
    assert normalized not in {"1130", "A320-200", "JFK", "Israir Airlines"}


def test_current_field_families_are_disjoint_and_exhaustive():
    family_values = tuple(FIELD_FAMILIES.values())
    assigned = [field for family in family_values for field in family]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == {
        "REGISTRATION", "SERIAL_NUMBER", "ICAO_ADDRESS", "SQUAWK",
        "BAROMETRIC_ALTITUDE", "VERTICAL_SPEED", "GPS_ALTITUDE", "TRACK",
        "GROUND_SPEED", "TRUE_AIRSPEED", "INDICATED_AIRSPEED", "MACH",
        "WIND", "TEMPERATURE", "FIR_UIR", "LATITUDE", "LONGITUDE", "AGE",
        "AIRCRAFT_CATEGORY", "AIRCRAFT_TYPE", "COUNTRY_OF_REGISTRATION",
        "SCHEDULED", "ACTUAL", "ESTIMATED", "airline_name", "primary_flight_id",
        "alternate_flight_id", "aircraft_code", "origin_airport_code", "origin_city",
        "origin_timezone_text", "destination_airport_code", "destination_city",
        "destination_timezone_text", "origin_scheduled_time", "origin_actual_time",
        "destination_scheduled_time", "destination_estimated_or_actual_time",
        "scroll_observed", "same_aircraft_throughout",
    }
    assert IDENTIFIER_FIELDS <= FIELD_FAMILIES["identifier"]
    assert NUMERIC_FIELDS <= FIELD_FAMILIES["numeric"]
    assert HEADING_FIELDS <= FIELD_FAMILIES["heading"]
    assert BOOLEAN_FIELDS <= FIELD_FAMILIES["boolean"]
    assert TEXT_FIELDS <= FIELD_FAMILIES["text"]
    assert TIME_FIELDS <= FIELD_FAMILIES["time"]


def test_comparison_agrees_on_identical_raw_values():
    result = compare_field_observations(
        "REGISTRATION", ocr_observation(), vlm_observation()
    )
    assert result.status == "agreement"
    assert result.ocr_normalized_value == result.vlm_normalized_value == "4X-ABC"
    assert result.resolved_value == "4X-ABC"
    assert result.diagnostics == ("values match after normalization",)


def test_comparison_agrees_after_identifier_and_track_normalization():
    identifier = compare_field_observations(
        "REGISTRATION", ocr_observation(" 4x-bal "), vlm_observation("4X-BAL")
    )
    track_ocr = ocr_observation("281ֲ°")
    track_vlm = vlm_observation("281°")
    heading = compare_field_observations("TRACK", track_ocr, track_vlm)
    assert identifier.status == "agreement"
    assert identifier.ocr_normalized_value == identifier.vlm_normalized_value == "4X-BAL"
    assert heading.status == "agreement"
    assert heading.ocr_normalized_value == heading.vlm_normalized_value == "281"


def test_comparison_conflicts_when_normalized_values_differ():
    result = compare_field_observations(
        "SERIAL_NUMBER", ocr_observation("1100"), vlm_observation("1130")
    )
    assert result.status == "conflict"
    assert result.ocr_normalized_value == "1100"
    assert result.vlm_normalized_value == "1130"
    assert result.resolved_value is None
    assert result.diagnostics == ("values differ after normalization",)


def test_ocr_only_preserves_vlm_unavailable_and_resolves_to_ocr_normalized_value():
    unavailable_vlm = vlm_observation(None, status="unavailable")
    result = compare_field_observations("REGISTRATION", ocr_observation(), unavailable_vlm)
    assert result.status == "ocr_only"
    assert result.ocr_observation is not None
    assert result.vlm_observation is unavailable_vlm
    assert result.ocr_normalized_value == "4X-ABC"
    assert result.vlm_normalized_value is None
    assert result.resolved_value == "4X-ABC"
    assert result.diagnostics == ("only OCR produced a usable value",)


def test_vlm_only_and_both_unavailable():
    unavailable_ocr = ocr_observation(include_value=False)
    vlm = vlm_observation()
    only_vlm = compare_field_observations("REGISTRATION", unavailable_ocr, vlm)
    neither = compare_field_observations("REGISTRATION", None, None)
    assert only_vlm.status == "vlm_only"
    assert only_vlm.ocr_observation is unavailable_ocr
    assert only_vlm.ocr_normalized_value is None
    assert only_vlm.resolved_value == "4X-ABC"
    assert only_vlm.diagnostics == ("only VLM produced a usable value",)
    assert neither.status == "unavailable"
    assert neither.resolved_value is None
    assert neither.diagnostics == ("neither source produced a usable value",)


def test_ambiguous_source_observations_are_preserved_but_not_available():
    ambiguous_vlm = vlm_observation("maybe", status="ambiguous")
    with_ocr = compare_field_observations("REGISTRATION", ocr_observation(), ambiguous_vlm)
    unavailable_ocr = replace(ocr_observation(include_value=False), status="ambiguous")
    neither = compare_field_observations("REGISTRATION", unavailable_ocr, ambiguous_vlm)
    ambiguous_ocr = replace(ocr_observation(), status="ambiguous", selected_candidate=None)
    with_vlm = compare_field_observations(
        "REGISTRATION", ambiguous_ocr, vlm_observation()
    )
    assert with_ocr.status == "ocr_only"
    assert with_ocr.vlm_observation is ambiguous_vlm
    assert with_ocr.vlm_normalized_value is None
    assert neither.status == "unavailable"
    assert neither.ocr_observation is unavailable_ocr
    assert neither.vlm_observation is ambiguous_vlm
    assert with_vlm.status == "vlm_only"
    assert with_vlm.ocr_observation is ambiguous_ocr
    assert with_vlm.resolved_value == "4X-ABC"


@pytest.mark.parametrize("ocr_value,vlm_value,expected", [("False", False, "agreement"), ("False", True, "conflict")])
def test_boolean_false_is_available_and_compared_explicitly(ocr_value, vlm_value, expected):
    result = compare_field_observations(
        "scroll_observed",
        ocr_observation(ocr_value),
        vlm_observation(vlm_value, field_name="scroll_observed"),
    )
    assert result.status == expected
    assert result.ocr_normalized_value is False
    assert result.vlm_normalized_value is vlm_value
    assert result.resolved_value is False if expected == "agreement" else result.resolved_value is None


def test_zero_and_leading_zero_identifiers_are_available_values():
    zero = compare_field_observations(
        "BAROMETRIC_ALTITUDE", ocr_observation("0"), vlm_observation("0")
    )
    identifier = compare_field_observations(
        "SQUAWK", ocr_observation("0012"), vlm_observation("0012")
    )
    assert zero.status == identifier.status == "agreement"
    assert zero.resolved_value == "0"
    assert identifier.resolved_value == "0012"


def test_comparison_preserves_source_records_and_separate_normalized_values():
    ocr = ocr_observation(" 4x-bal ")
    vlm = vlm_observation("4X-BAL")
    before_ocr = ocr.raw_value_text
    before_vlm = vlm.raw_value
    result = compare_field_observations("REGISTRATION", ocr, vlm)
    assert result.ocr_observation is ocr
    assert result.vlm_observation is vlm
    assert result.ocr_normalized_value == result.vlm_normalized_value == "4X-BAL"
    assert ocr.raw_value_text == before_ocr == " 4x-bal "
    assert vlm.raw_value == before_vlm == "4X-BAL"


def test_phase_65c_never_emits_temporal_mismatch_for_different_track_values():
    result = compare_field_observations(
        "TRACK", ocr_observation("100°"), vlm_observation("101°")
    )
    assert result.status == "conflict"
    assert result.status != "temporal_mismatch"


def test_temporal_field_sets_are_disjoint_and_cover_current_schema():
    assert STATIC_FIELDS.isdisjoint(DYNAMIC_FIELDS)
    assert STATIC_FIELDS | DYNAMIC_FIELDS == {
        "REGISTRATION", "SERIAL_NUMBER", "ICAO_ADDRESS", "SQUAWK",
        "BAROMETRIC_ALTITUDE", "VERTICAL_SPEED", "GPS_ALTITUDE", "TRACK",
        "GROUND_SPEED", "TRUE_AIRSPEED", "INDICATED_AIRSPEED", "MACH", "WIND",
        "TEMPERATURE", "FIR_UIR", "LATITUDE", "LONGITUDE", "AGE",
        "AIRCRAFT_CATEGORY", "AIRCRAFT_TYPE", "COUNTRY_OF_REGISTRATION",
        "SCHEDULED", "ACTUAL", "ESTIMATED", "airline_name", "primary_flight_id",
        "alternate_flight_id", "aircraft_code", "origin_airport_code", "origin_city",
        "origin_timezone_text", "destination_airport_code", "destination_city",
        "destination_timezone_text", "origin_scheduled_time", "origin_actual_time",
        "destination_scheduled_time", "destination_estimated_or_actual_time",
        "scroll_observed", "same_aircraft_throughout",
    }
    assert "SQUAWK" in DYNAMIC_FIELDS
    assert {"REGISTRATION", "AIRCRAFT_TYPE", "origin_airport_code"} <= STATIC_FIELDS
    assert {"TRACK", "LATITUDE", "LONGITUDE", "origin_actual_time"} <= DYNAMIC_FIELDS


def test_static_field_conflict_remains_unchanged():
    conflict = compare_field_observations(
        "REGISTRATION", ocr_observation("4X-ABC"), vlm_observation("4X-ABD")
    )
    adjusted = apply_temporal_awareness(conflict)
    assert conflict.status == adjusted.status == "conflict"
    assert adjusted is conflict


def test_dynamic_conflict_with_point_ocr_and_whole_clip_vlm_becomes_temporal_mismatch():
    ocr = ocr_observation("278ֲ°", timestamp=2.1)
    vlm = vlm_observation("281ֲ°")
    baseline = compare_field_observations("TRACK", ocr, vlm)
    result = apply_temporal_awareness(baseline)
    assert baseline.status == "conflict"
    assert result.status == "temporal_mismatch"
    assert result.resolved_value is None
    assert result.ocr_observation is ocr
    assert result.vlm_observation is vlm
    assert result.ocr_normalized_value == "278"
    assert result.vlm_normalized_value == "281"
    assert result.diagnostics[-1] == (
        "dynamic field values differ at non-comparable source times"
    )


def test_dynamic_conflict_with_matching_field_specific_point_times_remains_conflict():
    ocr = ocr_observation("278ֲ°", timestamp=4.0)
    vlm = vlm_observation("281ֲ°", evidence_time_s=4.0)
    baseline = compare_field_observations("TRACK", ocr, vlm)
    assert is_temporally_comparable("TRACK", ocr, vlm)
    assert apply_temporal_awareness(baseline) is baseline
    assert baseline.status == "conflict"
    assert baseline.resolved_value is None


@pytest.mark.parametrize(
    "baseline",
    [
        compare_field_observations("REGISTRATION", ocr_observation(), vlm_observation()),
        compare_field_observations("REGISTRATION", ocr_observation(), None),
        compare_field_observations(
            "REGISTRATION", ocr_observation(include_value=False), vlm_observation()
        ),
        compare_field_observations("REGISTRATION", None, None),
    ],
)
def test_non_conflict_statuses_are_returned_unchanged(baseline):
    assert apply_temporal_awareness(baseline) is baseline


def test_comparison_remains_time_agnostic_for_dynamic_values():
    ocr = ocr_observation("278ֲ°", timestamp=2.1)
    vlm = vlm_observation("281ֲ°", evidence_time_s=8.0)
    baseline = compare_field_observations("TRACK", ocr, vlm)
    assert baseline.status == "conflict"
    assert baseline.status != "temporal_mismatch"
    assert apply_temporal_awareness(baseline).status == "temporal_mismatch"


