"""Tests for field-level display behavior shared by future diary renderers."""

import subprocess
import sys

import pytest

from events_diary.canonical_events import (
    CanonicalDataQuality,
    CanonicalEvent,
    CanonicalFlight,
    CanonicalMovement,
    CanonicalRoute,
)
from events_diary.renderer_core import (
    FIELD_LABELS,
    NOT_AVAILABLE_DISPLAY,
    OMISSION_MARKER,
    UNRESOLVED_DISPLAY,
    display_value,
    format_investigation_interval,
    format_video_timestamp,
    label_for_path,
)


def _event(
    *,
    quality: CanonicalDataQuality | None = None,
    direction: str | None = "East",
    visual_direction: str | None = "northwest",
    track: float | None = 293.0,
    route_description: str | None = "Aircraft heading northwest over sea.",
    airline: str | None = "Example Air",
    origin: str | None = "ABC",
) -> CanonicalEvent:
    """Build one small canonical event with independently variable states."""
    return CanonicalEvent(
        event_id=0,
        investigation_id=4,
        event_time_s=1.6,
        start_time_s=1.6,
        end_time_s=4.9,
        flight=CanonicalFlight(primary_id="EXA123", airline=airline),
        route=CanonicalRoute(origin_airport_code=origin),
        movement=CanonicalMovement(
            track_degrees=track,
            direction=direction,
            visual_flight_direction=visual_direction,
            visual_route_description=route_description,
        ),
        data_quality=quality or CanonicalDataQuality(),
    )


def test_value_state_displays_canonical_value():
    assert display_value(_event(), "flight.primary_id") == "EXA123"


def test_unresolved_state_displays_human_label_not_internal_state():
    event = _event(quality=CanonicalDataQuality(
        unresolved_fields=("movement.direction",),
    ), direction=None)
    shown = display_value(event, "movement.direction")
    assert shown == UNRESOLVED_DISPLAY
    assert "unresolved" not in shown.casefold()


def test_not_available_state_displays_na():
    event = _event(quality=CanonicalDataQuality(
        not_available_fields=("route.origin_airport_code",),
    ), origin=None)
    assert display_value(event, "route.origin_airport_code") == "N/A"


def test_absent_field_is_omitted_not_displayed_as_not_available():
    event = _event(airline=None)
    assert event.field_state("flight.airline") == "absent"
    assert display_value(event, "flight.airline") is OMISSION_MARKER
    assert display_value(event, "flight.airline") != NOT_AVAILABLE_DISPLAY


def test_labels_are_human_readable_and_cover_renderer_paths_only():
    expected = {
        "event_time_s": "Event time",
        "flight.primary_id": "Flight",
        "flight.alternate_id": "Alternate flight",
        "flight.airline": "Airline",
        "route.origin_airport_code": "Origin",
        "route.destination_airport_code": "Destination",
        "route.scheduled_departure": "Scheduled departure",
        "route.actual_departure": "Actual departure",
        "route.scheduled_arrival": "Scheduled arrival",
        "route.estimated_or_actual_arrival": "Estimated/actual arrival",
        "aircraft.registration": "Registration",
        "aircraft.code": "Aircraft code",
        "aircraft.type": "Aircraft type",
        "movement.track_degrees": "Track",
        "movement.direction": "Direction",
        "movement.visual_flight_direction": "Visual direction",
        "movement.visual_route_description": "Visual route",
        "movement.visual_departure_context": "Departure context",
        "investigation_id": "Investigation",
        "start_time_s": "Start time",
        "end_time_s": "End time",
        "data_quality.unresolved_fields": "Unresolved fields",
        "data_quality.not_available_fields": "Not available fields",
    }
    assert dict(FIELD_LABELS) == expected
    assert all(label_for_path(path) == label for path, label in expected.items())
    assert all(path not in label for path, label in expected.items())


def test_label_api_rejects_unknown_and_internal_paths():
    for path in ("TRACK", "ocr.tokens", "movement.raw_vlm_direction"):
        with pytest.raises(ValueError, match="approved canonical renderer field"):
            label_for_path(path)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    ((0.0, "00:00.0"), (1.6, "00:01.6"), (65.2, "01:05.2")),
)
def test_video_relative_timestamp_format(seconds: float, expected: str):
    assert format_video_timestamp(seconds) == expected


def test_investigation_interval_format_is_relative_and_deterministic():
    assert format_investigation_interval(1.6, 4.9) == "00:01.6–00:04.9"
    with pytest.raises(ValueError, match="end must not precede"):
        format_investigation_interval(4.9, 1.6)


def test_track_degrees_adds_only_the_canonical_degree_unit():
    assert display_value(_event(), "movement.track_degrees") == "293°"


def test_strings_and_visual_route_claims_remain_source_faithful():
    route_text = "  Northwest; route claim unchanged.  "
    event = _event(airline="A B C Airlines", route_description=route_text)
    assert display_value(event, "flight.airline") == "A B C Airlines"
    assert display_value(event, "movement.visual_route_description") == route_text


def test_quality_list_values_show_labels_instead_of_canonical_paths():
    event = _event(quality=CanonicalDataQuality(
        unresolved_fields=("movement.direction", "movement.track_degrees"),
    ), direction=None, track=None)
    shown = display_value(event, "data_quality.unresolved_fields")
    assert shown == "Direction, Track"
    assert "movement." not in shown
    assert display_value(event, "data_quality.not_available_fields") is None


def test_renderer_core_import_does_not_load_phase6_modules():
    code = """
import sys
import events_diary.renderer_core
forbidden = {
    'events_diary.investigation_record', 'events_diary.fusion',
    'events_diary.ocr_fields', 'events_diary.vlm_review',
}
assert forbidden.isdisjoint(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
