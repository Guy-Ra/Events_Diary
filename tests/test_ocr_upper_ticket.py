"""FR24 header/route semantics, conservative abstention and lower-field preservation."""

from dataclasses import FrozenInstanceError, replace

import pytest

from events_diary.config import ROI
from events_diary.ocr_fields import extract_ocr_fields
from events_diary.ocr_upper_ticket import (
    AIRPORT_FIELDS,
    HEADER_FIELDS,
    extract_fr24_ticket,
    extract_fr24_tickets_batch,
    extract_fr24_upper_ticket,
)
from events_diary.ticket_ocr import OCRDetection, OCRTicketResult


def record(items=(), scale=1, x=16, y=121, candidate=5):
    """Build immutable source-frame detections without OCR/model execution."""
    roi = ROI(x, y, int(336 * scale), int(831 * scale))
    detections = tuple(OCRDetection(text, .91, tuple((x + a * scale, y + b * scale)
        for a, b in ((left, top), (left + width, top), (left + width, top + height), (left, top + height))))
        for text, (left, top, width, height) in items)
    return OCRTicketResult("test", "full_ticket_color_cubic_2x", roi, 2, detections, 0, candidate, 5, .5, "resolved")


HEADER = (("ISR972", (8, 29, 66, 18)), ("6H972", (83, 30, 41, 15)),
          ("A332", (133, 30, 34, 15)), ("Israir Airlines", (7, 52, 90, 16)),
          ("flightradar24", (254, 50, 77, 19)))
ROUTE = (("JFK", (50, 272, 66, 33)), ("TLV", (213, 272, 66, 33)),
         ("NEW YORK", (44, 313, 75, 15)), ("TELAVIV", (215, 313, 60, 15)),
         ("EDT (UTC-04:00)", (41, 332, 80, 14)), ("IDT (UTC +03:00)", (205, 332, 80, 14)),
         ("SCHEDULED", (8, 361, 70, 13)), ("2:30 pM", (98, 359, 60, 20)),
         ("SCHEDULED", (171, 361, 71, 13)), ("8:45 AM", (262, 358, 60, 21)),
         ("ACTUAL", (7, 394, 48, 14)), ("4:49 pM", (98, 392, 60, 21)),
         ("ESTIMATED", (171, 394, 67, 14)), ("9:31 AM", (262, 392, 60, 21)))


def test_real_shaped_header_route_raw_values_and_provenance():
    raw = record(HEADER + ROUTE)
    output = extract_fr24_ticket(raw)
    expected = dict(zip(HEADER_FIELDS + AIRPORT_FIELDS, ("Israir Airlines", "ISR972", "6H972", "A332",
        "JFK", "NEW YORK", "EDT (UTC-04:00)", "TLV", "TELAVIV", "IDT (UTC +03:00)"), strict=True))
    assert {f.field_name: f.raw_text for f in output.upper.fields} == expected
    assert output.upper.route.route_text == "JFK → TLV"
    assert output.upper.route.status == "extracted"
    for field in output.upper.fields:
        evidence = field.evidence[0]
        assert evidence.detection is raw.detections[evidence.detection_index]
        assert evidence.source_polygon is raw.detections[evidence.detection_index].source_polygon
        assert evidence.ocr_confidence == .91
    assert output.upper.provenance.actual_timestamp_seconds == .5
    assert output.upper.provenance.frame_index == 5
    assert output.fields == extract_ocr_fields(raw)
    assert [f.visual_side for f in output.time_fields] == ["left", "right", "left", "right"]
    assert [f.route_role for f in output.time_fields] == ["origin", "destination", "origin", "destination"]
    assert [f.field.raw_value_text for f in output.time_fields] == ["2:30 pM", "8:45 AM", "4:49 pM", "9:31 AM"]


@pytest.mark.parametrize("scale,x,y", [(1, 16, 121), (2, 32, 242), (.5, 0, 0), (1, 200, 600)])
def test_layout_uses_actual_crop_coordinates(scale, x, y):
    output = extract_fr24_upper_ticket(record(HEADER + ROUTE, scale, x, y))
    assert all(f.status == "extracted" for f in output.fields)
    assert output.route.route_text == "JFK → TLV"


@pytest.mark.parametrize("index", range(4))
def test_missing_header_token_does_not_shift_other_roles(index):
    items = HEADER[:index] + HEADER[index + 1:]
    output = extract_fr24_upper_ticket(record(items))
    name = ("primary_flight_id", "alternate_flight_id", "aircraft_code", "airline_name")[index]
    assert output.get(name).status == "unavailable"
    assert sum(f.status == "extracted" for f in output.fields) == 3


@pytest.mark.parametrize("primary,alternate,aircraft,airline", [("HFA853", "E2853", "AT76", "airHaifa"),
    ("PGT1719", "PC1719", "A320", "Pegasus"), ("ELY5425", "LY5425", "B738", "Sundor"),
    ("AHY022", "J222", "A21N", "Azerbaijan Airlines")])
def test_identity_shapes_do_not_use_known_airline_values(primary, alternate, aircraft, airline):
    values = (primary, alternate, aircraft, airline)
    items = tuple((value, box) for value, (_, box) in zip(values, HEADER[:4], strict=True))
    output = extract_fr24_upper_ticket(record(items))
    assert [output.get(n).raw_text for n in ("primary_flight_id", "alternate_flight_id", "aircraft_code", "airline_name")] == list(values)


def test_overlapping_identifier_shapes_do_not_shift_when_aircraft_code_is_missing():
    items = (("AHY022", (8, 29, 76, 18)), ("J222", (92, 30, 34, 15)))
    output = extract_fr24_upper_ticket(record(items))
    assert output.get("alternate_flight_id").raw_text == "J222"
    assert output.get("aircraft_code").status == "unavailable"


@pytest.mark.parametrize("index,name", [(0, "primary_flight_id"), (1, "alternate_flight_id"),
                                       (2, "aircraft_code"), (3, "airline_name")])
def test_competing_header_candidates_abstain(index, name):
    output = extract_fr24_upper_ticket(record(HEADER + (HEADER[index],)))
    field = output.get(name)
    assert field.status == "ambiguous" and field.raw_text is None and len(field.evidence) == 2


@pytest.mark.parametrize("index,name", [(0, "origin_airport_code"), (1, "destination_airport_code"),
    (2, "origin_city"), (3, "destination_city"), (4, "origin_timezone_text"), (5, "destination_timezone_text")])
def test_missing_route_components_are_not_inferred(index, name):
    output = extract_fr24_upper_ticket(record(HEADER + ROUTE[:index] + ROUTE[index + 1:]))
    assert output.get(name).status == "unavailable"
    assert output.get(name).raw_text is None
    assert output.route.status == ("unavailable" if index < 2 else "extracted")


@pytest.mark.parametrize("index,name", [(0, "origin_airport_code"), (2, "origin_city"), (4, "origin_timezone_text")])
def test_duplicate_route_candidates_remain_ambiguous(index, name):
    output = extract_fr24_upper_ticket(record(HEADER + ROUTE + (ROUTE[index],)))
    assert output.get(name).status == "ambiguous" and output.get(name).raw_text is None


def test_no_anchor_does_not_read_photo_or_lower_fields_as_route():
    output = extract_fr24_upper_ticket(record(HEADER + ROUTE[:6] + (("B738", (45, 520, 60, 16)),)))
    assert all(output.get(name).status == "unavailable" for name in AIRPORT_FIELDS)


def test_partial_scrolled_route_retains_timezone_only_and_never_photo_codes():
    shifted = tuple((value, (x, y - 59, w, h)) for value, (x, y, w, h) in ROUTE)
    output = extract_fr24_upper_ticket(record(HEADER + shifted))
    assert output.get("origin_airport_code").status == "unavailable"
    assert output.get("origin_city").status == "unavailable"
    assert output.get("origin_timezone_text").raw_text == "EDT (UTC-04:00)"
    assert output.route.route_text is None


def test_wrong_row_anchors_abstain_instead_of_composing_unrelated_route_blocks():
    shifted = tuple((value, (x, y + (35 if x > 168 else 0), w, h)) for value, (x, y, w, h) in ROUTE)
    output = extract_fr24_upper_ticket(record(HEADER + shifted))
    assert all(output.get(name).status == "ambiguous" for name in AIRPORT_FIELDS)


def test_multiple_scheduled_anchors_abstain_on_affected_side():
    output = extract_fr24_upper_ticket(record(HEADER + ROUTE + (("SCHEDULED", (8, 405, 70, 13)),)))
    assert output.get("origin_airport_code").status == "ambiguous"
    assert output.route.status == "ambiguous"


def test_aircraft_photo_and_operated_by_line_are_not_airline():
    items = HEADER + (("Other Airline", (20, 160, 100, 20)),
                      ("Operated by Klasjet", (8, 71, 130, 18)))
    assert extract_fr24_upper_ticket(record(items)).get("airline_name").raw_text == "Israir Airlines"


def test_crossing_time_label_has_explicit_unknown_side():
    output = extract_fr24_ticket(record((("SCHEDULED", (130, 361, 100, 13)),)))
    assert output.time_fields[0].visual_side == "unknown"
    assert output.time_fields[0].route_role is None
    assert output.upper.route.route_text is None


def test_empty_output_exposes_every_slot_and_explicit_unavailability():
    output = extract_fr24_ticket(record())
    assert len(output.upper.fields) == 10
    assert all(f.raw_text is None and f.status == "unavailable" for f in output.upper.fields)
    assert output.upper.route.status == "unavailable"
    assert output.fields == output.time_fields == ()
    with pytest.raises(KeyError):
        output.upper.get("invented")


def test_results_are_frozen_input_unchanged_and_batch_never_merges_views():
    raw = record(HEADER + ROUTE)
    snapshot = raw.detections
    first, second = extract_fr24_tickets_batch((raw, record(HEADER, candidate=33)))
    assert first == extract_fr24_ticket(raw)
    assert raw.detections is snapshot
    assert second.upper.provenance.candidate_index == 33
    assert second.upper.route.route_text is None
    assert first.upper.route.origin is first.upper.get("origin_airport_code")
    for item in (first, first.upper, first.upper.fields[0], first.upper.route, first.time_fields[0]):
        with pytest.raises(FrozenInstanceError):
            item.unexpected = 1
    assert extract_fr24_tickets_batch(()) == ()


@pytest.mark.parametrize("bad", [None, "abc", 3, {}, [None]])
def test_invalid_api_inputs(bad):
    with pytest.raises(TypeError):
        extract_fr24_ticket(bad)
    with pytest.raises(TypeError):
        extract_fr24_tickets_batch(bad)


@pytest.mark.parametrize("change", [{"actual_timestamp_seconds": float("nan")},
    {"actual_timestamp_seconds": -1}, {"resolution_status": "unresolved"}, {"frame_index": -1}])
def test_resolved_physical_provenance_is_required(change):
    with pytest.raises(ValueError):
        extract_fr24_ticket(replace(record(), **change))


def test_ocr_order_is_not_semantic_order_and_raw_whitespace_is_preserved():
    items = tuple(("  " + value + "  ", box) for value, box in reversed(HEADER + ROUTE))
    output = extract_fr24_ticket(record(items))
    assert output.upper.get("primary_flight_id").raw_text == "  ISR972  "
    assert output.upper.get("destination_city").raw_text == "  TELAVIV  "
    assert output.upper.route.route_text == "  JFK   →   TLV  "
